#!/usr/bin/env bash
set -euo pipefail

# === CONFIGURATION ===
CLUSTER_NAME="pipeline_unsloth"
YAML_FILE="serving.yaml"
CONFIG_FILE="${CONFIG_FILE:-config/example_config.json}"
KEY_PATH="$HOME/.ssh/id_ed25519.pub"     
POLL_INTERVAL=10                         
TIMEOUT=600                              
SYNC_INTERVAL=30                         
REMOTE_OUTPUT_DIR="/root/sky_workdir/outputs"             
LOCAL_OUTPUT_DIR="./outputs"

# === STEP 0: Merge JSON config into YAML file ===
echo "[INFO] Merging JSON config ($CONFIG_FILE) into YAML file ($YAML_FILE)..."

# Check if yq is installed
if ! command -v yq &> /dev/null; then
    echo "[ERROR] yq is not installed. Please install yq to use JSON config."
    echo "[ERROR] Install with: brew install yq (on macOS) or visit https://github.com/mikefarah/yq"
    exit 1
fi

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if YAML file exists
if [[ ! -f "$YAML_FILE" ]]; then
    echo "[ERROR] YAML file not found: $YAML_FILE"
    exit 1
fi

# Merge JSON config into YAML using yq
yq eval '.envs = load("'"$CONFIG_FILE"'").envs | .resources = load("'"$CONFIG_FILE"'").resources | .file_mounts = load("'"$CONFIG_FILE"'").file_mounts | .workdir = load("'"$CONFIG_FILE"'").workdir' -P -i "$YAML_FILE"

if [[ $? -eq 0 ]]; then
    echo "[INFO] ✅ Successfully merged config from $CONFIG_FILE into $YAML_FILE"
else
    echo "[ERROR] Failed to merge config into YAML file"
    exit 1
fi

# === STEP 1: Launch cluster asynchronously ===
echo "[INFO] Launching SkyPilot cluster: $CLUSTER_NAME"
sky launch "$YAML_FILE" -c "$CLUSTER_NAME" --yes &
LAUNCH_PID=$!

# === STEP 2: Wait for Vast.ai instance to appear ===
echo "[INFO] Waiting for Vast.ai instance to start running..."
START_TIME=$(date +%s)
INSTANCE_ID=""

while true; do
    CURRENT_TIME=$(date +%s)
    if (( CURRENT_TIME - START_TIME > TIMEOUT )); then
        echo "[ERROR] Timeout waiting for Vast.ai instance."
        kill $LAUNCH_PID 2>/dev/null || true
        exit 1
    fi

    INSTANCE_ID=$(vastai show instances | awk '/running/ {print $1}' | tail -n 1 || true)
    if [[ -n "$INSTANCE_ID" ]]; then
        echo "[INFO] Detected running Vast.ai instance: $INSTANCE_ID"
        break
    fi

    echo "[INFO] No running instance yet... retrying in ${POLL_INTERVAL}s"
    sleep "$POLL_INTERVAL"
done

# === STEP 3: Attach SSH key ===
echo "[INFO] Attaching SSH key to instance $INSTANCE_ID..."
if vastai attach ssh "$INSTANCE_ID" "$KEY_PATH"; then
    echo "[INFO] ✅ SSH key successfully attached to instance $INSTANCE_ID."
else
    echo "[WARN] ❌ Failed to attach SSH key."
fi

# === STEP 4: Wait for SkyPilot launch to finish ===
wait "$LAUNCH_PID"
echo "[INFO] ✅ SkyPilot launch complete. Instance ready with attached SSH key."

# === STEP 5: Get cluster IP ===
echo "[INFO] Detecting cluster IP..."
SSH_ADDR=$(sky status --ip "$CLUSTER_NAME" | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}')
if [[ -z "$SSH_ADDR" ]]; then
    echo "[ERROR] Could not determine cluster IP. Run 'sky status' manually to check."
    exit 1
fi
echo "[INFO] Cluster IP: $SSH_ADDR"

# === STEP 6: Start one-way sync ===
echo "[INFO] 🔁 Starting one-way rsync from $REMOTE_OUTPUT_DIR to $LOCAL_OUTPUT_DIR"
mkdir -p "$LOCAL_OUTPUT_DIR"

(
  while true; do
    echo "[SYNC] Pulling latest files..."
    rsync -avz -e "ssh -p 21080 -i $HOME/.ssh/id_ed25519 -o StrictHostKeyChecking=no" \
      root@"$SSH_ADDR":"$REMOTE_OUTPUT_DIR"/ "$LOCAL_OUTPUT_DIR"/
    echo "[SYNC] ✅ Sync complete. Sleeping for $SYNC_INTERVAL seconds..."
    sleep "$SYNC_INTERVAL"
  done
) &

SYNC_PID=$!
trap "echo '[INFO] Stopping live sync...'; kill $SYNC_PID 2>/dev/null || true" EXIT

# === STEP 7: Keep process alive ===
echo "[INFO] 🟢 Live sync active. Press Ctrl+C to stop."
wait