import os
from typing import Dict, List

import hydra
import requests
import wandb
from accelerate import Accelerator
from datasets import Dataset, load_dataset, load_from_disk
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
from ainfinity.utils import settings

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


def load_datalog_dataset(dataset_args: DictConfig) -> DatasetDict:
    """
    Load datalog dataset from URLs according to YAML config.
    Always returns a DatasetDict with 'train' and 'validation' splits.

    Supports:
    - schema_type: conversation, text_generation, etc.
    - urls: list of URLs to download JSONL files
    - shuffle: whether to shuffle the combined dataset
    - train_eval_split: ratio for train/validation split (default 0.8)
    - train_max_samples: max samples for training set
    - validation_max_samples: max samples for validation set
    """
    import json

    print(f"Loading datalog dataset from {len(dataset_args.urls)} URLs...")

    # Download and parse JSONL files from URLs
    all_examples: List[Dict] = []
    for url in dataset_args.urls:
        print(f"Downloading from: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Parse JSONL
        for line in response.text.strip().split("\n"):
            if line.strip():
                example = json.loads(line)
                all_examples.append(example)

    print(f"Loaded {len(all_examples)} examples total")

    # Create dataset from examples
    dataset = Dataset.from_list(all_examples)

    # Shuffle if requested
    if dataset_args.get("shuffle", True):
        dataset = dataset.shuffle(seed=42)

    # Split into train/validation
    train_eval_split = dataset_args.get("train_eval_split", 0.8)
    split_dataset = dataset.train_test_split(train_size=train_eval_split, seed=42)

    train_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"]

    # Limit samples if requested
    train_max_samples = dataset_args.get("train_max_samples", None)
    if train_max_samples is not None and train_max_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(train_max_samples))

    validation_max_samples = dataset_args.get("validation_max_samples", None)
    if validation_max_samples is not None and validation_max_samples < len(validation_dataset):
        validation_dataset = validation_dataset.select(range(validation_max_samples))

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(validation_dataset)}")

    return DatasetDict({"train": train_dataset, "validation": validation_dataset})


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


def load_dataset_wrapper(dataset_args: DictConfig) -> tuple[DatasetDict, str]:
    """
    Unified dataset loader that routes to appropriate loader based on source.
    Returns (dataset, text_column_name)
    """
    # DataSource enum values from YAML: "datalog" or "huggingface"
    source = dataset_args.get("source", "huggingface")

    if source == "datalog":  # DataSource.DATALOG.value
        print("Loading from datalog source...")
        dataset = load_datalog_dataset(dataset_args)
        text_col = "messages"  # Datalog uses 'messages' field for conversation format
    else:  # DataSource.HUGGINGFACE.value or default
        print("Loading from HuggingFace source...")
        dataset = load_hf_dataset(dataset_args)
        text_col = dataset_args.text_col

    return dataset, text_col


@hydra.main(config_path="config/yaml", config_name="finetuning", version_base=None)
def main(config: DictConfig):
    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        wandb.login(key=settings.WANDB_API_KEY)  # type: ignore[attr-defined]

        wandb.init(project=settings.PROJECT, job_type="training", anonymous="allow")  # type: ignore[attr-defined]
    accelerator.wait_for_everyone()

    model_args = config.model
    dataset_args = config.dataset
    training_args = TrainingArguments(**config.training_arguments)

    training_args = configure_hub_settings(training_args)

    set_seed(config.seed)
    ds_cached_dir = dataset_args.get("cache_dir", "./data_cache_dir")
    model_cached_dir = model_args.get("cache_dir", "./model_cache_dir")

    if accelerator.is_main_process:
        dataset, text_col = load_dataset_wrapper(dataset_args)

        tokenizer = load_tokenizer(model_args)
        model = load_model(model_args, config.training_arguments, tokenizer)

        def apply_chat_template(example):
            output = tokenizer.apply_chat_template(
                example[text_col],
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

    # Get train/validation datasets based on source
    source = dataset_args.get("source", "huggingface")
    if source == "datalog":  # DataSource.DATALOG.value
        train_dataset = formated_dataset["train"]
        eval_dataset = formated_dataset["validation"]
    else:  # DataSource.HUGGINGFACE.value or default
        train_dataset = formated_dataset[dataset_args.split.train.name]
        eval_dataset = formated_dataset[dataset_args.split.validation.name]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
