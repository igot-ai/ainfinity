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
