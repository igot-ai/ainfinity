
# ainfinity

A flexible framework for fine-tuning and evaluating large language models (LLMs) with DeepSpeed, Accelerate, and HuggingFace Transformers. Designed for research and production, it supports custom datasets, efficient training, and experiment tracking.

## Features

- **Fine-tuning**: Easily fine-tune LLMs using DeepSpeed and Accelerate.
- **Configurable**: YAML-based configuration for datasets, models, and training.
- **Sample Datasets**: Includes formats like Alpaca, ShareGPT, and more.
- **Experiment Tracking**: Integrated with Weights & Biases (wandb).
- **Extensible**: Modular codebase for adding new models, datasets, or evaluation logic.

## Project Structure

```
ainfinity/
  config.py            # Project settings and environment variables
  datasample/          # Example datasets (Alpaca, ShareGPT, etc.)
  eval/                # Evaluation utilities (WIP)
  schema/              # Data schemas (WIP)
  serving/             # Model serving (WIP)
  train/
	 dataset.py         # Dataset loading utilities
	 finetune.py        # Main fine-tuning script
	 helper.py          # Training helpers
	 config/            # Training/config YAMLs
	 hparams/           # Hyperparameter configs
	 script/            # (Reserved for training scripts)
assets/                # (Reserved for assets)
data/                  # (Reserved for user data)
finetuning-checkpoints/# Model checkpoints
script/
  finetune/
	 finetune_with_deepspeed.sh # Example shell script to launch training
tests/                 # Test scripts
```


## Installation

1. **Clone the repository**
	```sh
	git clone <repo-url>
	cd ainfinity
	```

2. **Install [uv](https://github.com/astral-sh/uv)** (if not already installed):
	```sh
	curl -Ls https://astral.sh/uv/install.sh | sh
	```

3. **Install dependencies with uv**
	Requires Python 3.12+.
	```sh
	uv pip install -e .
	```

	For GPU support:
	```sh
	uv pip install '.[gpu]'
	```

4. **Set up environment variables**
	Copy `.env.example` to `.env` and fill in your HuggingFace and wandb tokens.

## Optional: Install flash-attn

To install [flash-attn](https://github.com/Dao-AILab/flash-attention) with parallel build jobs:

```sh
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

## Usage

### Fine-tuning

Edit the YAML configs in `ainfinity/train/config/yaml/` as needed.

Run fine-tuning with DeepSpeed:
```sh
bash script/finetune/finetune_with_deepspeed.sh
```

### Custom Datasets

Add your dataset in `ainfinity/datasample/` and update the config YAMLs.

## Configuration

- **Model config**: `ainfinity/train/config/yaml/model/qwen3.yaml`
- **Dataset config**: `ainfinity/train/config/yaml/dataset/orca_chat.yaml`
- **Training config**: `ainfinity/train/config/yaml/finetuning.yaml`

## Contributing

Contributions are welcome! Please open issues or pull requests.

## License

See [LICENSE](LICENSE).
