import os

import hydra
import wandb
from accelerate import Accelerator
from datasets import load_from_disk
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

from ainfinity.core.data_collator import CompletionOnlyLMWithPaddingFree
from ainfinity.core.dataset import load_dataset_wrapper
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
                padding=model_args.padding,
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

    # Get padding_free setting from config, default to False
    compute_settings = select_torch_compute(
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        attn_impl=model_args.attn_implementation,
    )
    attn_impl = compute_settings.attn_impl
    if attn_impl == "flash_attention_2":
        padding_free = True
    else:
        padding_free = False

    collator = CompletionOnlyLMWithPaddingFree(
        instruction_template=model_args.instruction_template,
        response_template=model_args.response_template,
        tokenizer=tokenizer,
        mlm=False,
        padding_free=padding_free,
        return_flash_attn_kwargs=padding_free,  # Enable Flash Attention kwargs when padding_free
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
