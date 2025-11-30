#!/bin/bash

cd "$(dirname "$0")/../../" || exit

ACCELERATE_CONFIG="./ainfinity/train/config/accelerate_config/ds2_config.yaml"

PYTHON_FILE="./ainfinity/train/finetune.py"

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  "$PYTHON_FILE"
