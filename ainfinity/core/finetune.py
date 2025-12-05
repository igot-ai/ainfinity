import os

import hydra
import wandb
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM

from ainfinity.core.helper import (
    configure_hub_settings,
    print_trainable_parameters,
    select_torch_compute,
)
from ainfinity.utils import logger, settings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def load_tokenizer(model_args: DictConfig) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast,
        return_tensors="pt",
        revision=model_args.revision,
        token=settings.HF_TOKEN,
        padding_side=model_args.padding_side,
        truncation_side=model_args.truncation_side,
        truncation=True,
        padding=True,
        max_length=model_args.model_max_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_args: DictConfig,
    training_args: DictConfig,
    tokenizer: PreTrainedTokenizerFast,
):
    config = AutoConfig.from_pretrained(
        model_args.pretrained_model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.revision,
        token=settings.HF_TOKEN,
    )
    config.use_cache = False

    compute_settings = select_torch_compute(
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        attn_impl=model_args.attn_implementation,
    )
    dtype, attn_impl = compute_settings.dtype, compute_settings.attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.pretrained_model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        token=settings.HF_TOKEN,
        config=config,
        revision=model_args.revision,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()

    model.config.use_cache = False

    print_trainable_parameters(model)
    return model


def load_hf_dataset(dataset_args: DictConfig) -> DatasetDict:
    """
    Load HuggingFace dataset according to YAML config.
    Always returns a DatasetDict.

    Supports:
    - empty split configs (validation:)
    - shuffle, max_samples
    - revision
    """
    dataset = load_dataset(
        dataset_args.name,
        split=None,
        revision=dataset_args.get("revision", None),
    )

    # Prepare final splits according to config
    dataset_splits = {}
    split_cfg = dataset_args.get("split", {})

    for split_key, s_cfg in split_cfg.items():
        if s_cfg is not None:
            print(f"split_key={split_key}, s_cfg={s_cfg}")
            split_name = s_cfg.get("name", split_key)
            subset = dataset[split_name]

            # Shuffle if requested
            if s_cfg.get("shuffle", False):
                subset = subset.shuffle(seed=42)

            # Limit number of samples if requested
            max_samples = s_cfg.get("max_samples", None)
            if max_samples is not None:
                subset = subset.select(range(max_samples))

            dataset_splits[split_key] = subset

    dataset_dict = DatasetDict(dataset_splits)

    return dataset_dict


@hydra.main(config_path="config/yaml", config_name="finetuning", version_base=None)
def main(config: DictConfig):
    accelerator = Accelerator()

    # Get run_id from config, environment variable, or generate one
    run_id = config.get("run_id", None) or os.getenv("WANDB_RUN_ID", None)

    if accelerator.is_local_main_process:
        wandb.login(key=settings.WANDB_API_KEY)  # type: ignore[attr-defined]

        # Initialize WandB with the run_id
        run = wandb.init(
            project=settings.PROJECT,
            job_type="finetuning",
            id=run_id,  # Use provided run_id or let WandB auto-generate
            resume="allow",  # Allow resuming if this run_id exists
            name=config.get("run_name", None),  # Optional custom name
            tags=config.get("tags", "finetuning"),  # Optional tags
        )  # type: ignore[attr-defined]

        # Log the run information
        logger.info("=" * 80)
        logger.info("🚀 WandB Run Information")
        logger.info("=" * 80)
        logger.info(f"Run ID: {run.id}")
        logger.info(f"Run Name: {run.name}")
        logger.info(f"Run URL: {run.url}")
        logger.info(f"Project: {run.project}")
        logger.info(f"Entity: {run.entity}")
        logger.info(f"Full Path: {run.entity}/{run.project}/{run.id}")
        logger.info("=" * 80)

    accelerator.wait_for_everyone()

    logger.info("Starting training...")
    model_args = config.model
    dataset_args = config.dataset
    training_args = TrainingArguments(**config.training_arguments)

    training_args = configure_hub_settings(training_args)

    set_seed(config.seed)
    ds_cached_dir = dataset_args.get("cache_dir", "./data_cache_dir")
    model_cached_dir = model_args.get("cache_dir", "./model_cache_dir")

    if accelerator.is_main_process:
        dataset = load_hf_dataset(dataset_args)

        tokenizer = load_tokenizer(model_args)
        model = load_model(model_args, config.training_arguments, tokenizer)

        def apply_chat_template(example):
            output = tokenizer.apply_chat_template(
                example[dataset_args.text_col],
                add_generation_prompt=False,
                tokenize=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors=None,
                truncation=True,
                padding="longest",
                max_length=model_args.model_max_length,
            )

            return output

        formated_dataset = dataset.map(apply_chat_template, batched=False)
        formated_dataset.save_to_disk(ds_cached_dir)

        tokenizer.save_pretrained(model_cached_dir)
        model.save_pretrained(model_cached_dir)

    accelerator.wait_for_everyone()

    formated_dataset = load_from_disk(ds_cached_dir)

    if not accelerator.is_main_process:
        model_args.pretrained_model_name_or_path = model_cached_dir
    tokenizer = load_tokenizer(model_args)
    model = load_model(model_args, config.training_arguments, tokenizer)

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=model_args.instruction_template,
        response_template=model_args.response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    # # ========== DEBUGGING SECTION START ==========
    # This is due to eval loss is NaN!!!
    # logger.info("=" * 80)
    # logger.info("DEBUGGING TOKENIZATION AND DATA COLLATOR")
    # logger.info("=" * 80)

    # # Check instruction and response templates
    # logger.info(f"Instruction template: {model_args.instruction_template}")
    # logger.info(f"Response template: {model_args.response_template}")

    # # Tokenize the templates to see their token IDs
    # inst_tokens = tokenizer.encode(model_args.instruction_template, add_special_tokens=False)
    # resp_tokens = tokenizer.encode(model_args.response_template, add_special_tokens=False)
    # logger.info(f"Instruction template tokens: {inst_tokens}")
    # logger.info(f"Response template tokens: {resp_tokens}")

    # # Inspect a sample from training set
    # logger.info("\n" + "=" * 80)
    # logger.info("SAMPLE FROM TRAINING SET:")
    # logger.info("=" * 80)
    # train_sample = formated_dataset[dataset_args.split.train.name][0]
    # logger.info(f"Keys in sample: {train_sample.keys()}")
    # logger.info(f"Input IDs length: {len(train_sample['input_ids'])}")
    # logger.info(f"First 20 input IDs: {train_sample['input_ids'][:20]}")
    # logger.info(f"Last 20 input IDs: {train_sample['input_ids'][-20:]}")

    # decoded_text = tokenizer.decode(train_sample["input_ids"])
    # logger.info(f"\nDecoded text (first 500 chars):\n{decoded_text[:500]}")
    # logger.info(f"\nDecoded text (last 500 chars):\n{decoded_text[-500:]}")

    # # Check if response template appears in the decoded text
    # if model_args.response_template in decoded_text:
    #     logger.info(f"\n✓ Response template FOUND in decoded text")
    # else:
    #     logger.warning(f"\n✗ Response template NOT FOUND in decoded text")
    #     logger.warning(f"This will cause the collator to mask ALL tokens, resulting in NaN loss!")

    # # Inspect a sample from validation set
    # logger.info("\n" + "=" * 80)
    # logger.info("SAMPLE FROM VALIDATION SET:")
    # logger.info("=" * 80)
    # val_sample = formated_dataset[dataset_args.split.validation.name][0]
    # logger.info(f"Keys in sample: {val_sample.keys()}")
    # logger.info(f"Input IDs length: {len(val_sample['input_ids'])}")

    # decoded_val_text = tokenizer.decode(val_sample["input_ids"])
    # logger.info(f"\nDecoded text (first 500 chars):\n{decoded_val_text[:500]}")

    # if model_args.response_template in decoded_val_text:
    #     logger.info(f"\n✓ Response template FOUND in validation sample")
    # else:
    #     logger.warning(f"\n✗ Response template NOT FOUND in validation sample")

    # # Test the data collator
    # logger.info("\n" + "=" * 80)
    # logger.info("TESTING DATA COLLATOR:")
    # logger.info("=" * 80)

    # # Create a small batch with one sample
    # test_batch = [train_sample]
    # try:
    #     collated = collator(test_batch)
    #     logger.info(f"Collator output keys: {collated.keys()}")

    #     if "labels" in collated:
    #         labels = collated["labels"][0]  # First sample in batch
    #         logger.info(f"Labels shape: {labels.shape}")
    #         logger.info(f"Labels (first 20): {labels[:20]}")
    #         logger.info(f"Labels (last 20): {labels[-20:]}")

    #         # Count how many tokens are NOT masked (-100)
    #         non_masked = (labels != -100).sum().item()
    #         total = labels.shape[0]
    #         masked = (labels == -100).sum().item()

    #         logger.info(f"\nLabel statistics:")
    #         logger.info(f"  Total tokens: {total}")
    #         logger.info(f"  Masked tokens (ignore): {masked} ({100*masked/total:.1f}%)")
    #         logger.info(f"  Non-masked tokens (used for loss): {non_masked} ({100*non_masked/total:.1f}%)")

    #         if non_masked == 0:
    #             logger.error("\n" + "!" * 80)
    #             logger.error("ERROR: ALL TOKENS ARE MASKED!")
    #             logger.error("This will cause NaN loss during evaluation.")
    #             logger.error("The response_template is likely not matching the tokenized sequence.")
    #             logger.error("!" * 80)
    #         else:
    #             logger.info(f"\n✓ Found {non_masked} tokens for loss computation")
    #     else:
    #         logger.warning("No 'labels' key in collator output!")

    # except Exception as e:
    #     logger.error(f"Error testing collator: {e}")
    #     import traceback

    #     logger.error(traceback.format_exc())

    # logger.info("\n" + "=" * 80)
    # logger.info("END OF DEBUGGING SECTION")
    # logger.info("=" * 80 + "\n")
    # # ========== DEBUGGING SECTION END ==========

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formated_dataset[dataset_args.split.train.name],
        eval_dataset=formated_dataset[dataset_args.split.validation.name],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
