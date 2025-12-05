#!/bin/bash

cd "$(dirname "$0")/../../" || exit

ACCELERATE_CONFIG="./ainfinity/core/config/accelerate_config/ds2_config.yaml"

PYTHON_FILE="./ainfinity/core/finetune.py"

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  "$PYTHON_FILE"
