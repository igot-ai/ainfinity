import argparse
import logging

import torch
from datasets import load_dataset
from train.monitoring import get_tracker
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Unsloth Fine-tuning Script")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Job name for this fine-tuning job",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/llama-3-8b-bnb-4bit",
        help="Base model name",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset file (JSONL)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--project_name", type=str, default="slm-finetuning", help="WandB project name"
    )
    parser.add_argument(
        "--run_name", type=str, default="default-run", help="WandB run name"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Initialize Monitoring
    tracker = get_tracker(
        project_name=args.project_name, run_name=args.run_name, config=vars(args)
    )

    # 2. Load Model
    tracker.log_metrics({"status": "loading_model"}, step=0)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # 3. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # 4. Load Dataset
    # Load from local JSONL file
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Simple formatting function for instruction tuning
    # Assumes standard keys: instruction, context, response
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples.get(
            "context", [""] * len(instructions)
        )  # Handle missing context
        outputs = examples["response"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            texts.append(text)
        return {
            "text": texts,
        }

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    # 5. Train
    tracker.log_metrics({"status": "starting_training"}, step=0)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            report_to="wandb",
        ),
    )

    trainer_stats = trainer.train()

    tracker.log_metrics(
        {
            "train_runtime": trainer_stats.metrics["train_runtime"],
            "train_loss": trainer_stats.metrics["train_loss"],
        }
    )

    # 6. Save Model
    tracker.log_metrics({"status": "saving_model"}, step=0)

    # Create models directory and save with model_id
    import os

    model_save_path = os.path.join("models", args.model_id)
    os.makedirs(model_save_path, exist_ok=True)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    logger.info(f"Model saved to: {model_save_path}")

    # 7. Register model in WandB Model Registry
    tracker.log_metrics({"status": "registering_model"}, step=0)
    tracker.log_artifact(model_save_path, artifact_type="model")

    # Log model to WandB Model Registry with metadata
    import wandb

    if wandb.run:
        # Create a model artifact for the registry
        model_artifact = wandb.Artifact(
            name=f"finetuned-{args.model_id}",
            type="model",
            description=f"Fine-tuned {args.model_name} on custom dataset",
            metadata={
                "base_model": args.model_name,
                "epochs": args.epochs,
                "max_seq_length": args.max_seq_length,
                "dataset": args.dataset_path,
                "train_loss": trainer_stats.metrics.get("train_loss"),
                "train_runtime": trainer_stats.metrics.get("train_runtime"),
            },
        )
        model_artifact.add_dir(model_save_path)
        wandb.log_artifact(model_artifact)

        # Link to model registry
        wandb.run.link_artifact(model_artifact, f"model-registry/{args.model_id}")

        logger.info(f"Model registered in WandB: {model_artifact.name}")

    # 8. Update local registry with artifact path
    try:
        import json
        import fcntl
        import time

        registry_path = "train/services/registry.json"
        if os.path.exists(registry_path):
            # Use file locking to prevent race conditions
            max_retries = 5
            retry_delay = 0.5

            for attempt in range(max_retries):
                try:
                    with open(registry_path, "r+") as f:
                        # Acquire exclusive lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                        try:
                            # Read current data
                            f.seek(0)
                            registry_data = json.load(f)

                            # Update the specific model entry
                            if args.model_id in registry_data:
                                registry_data[args.model_id]["status"] = "completed"
                                registry_data[args.model_id]["artifact_path"] = (
                                    os.path.abspath(model_save_path)
                                )

                                # Write back
                                f.seek(0)
                                f.truncate()
                                json.dump(registry_data, f, indent=2)

                                logger.info(
                                    f"Updated registry for model {args.model_id}"
                                )
                                break
                        finally:
                            # Release lock
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except (IOError, OSError) as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Registry lock attempt {attempt + 1} failed, retrying..."
                        )
                        time.sleep(retry_delay)
                    else:
                        raise
    except Exception as e:
        logger.error(f"Failed to update registry: {e}")

    tracker.finish()


if __name__ == "__main__":
    main()
