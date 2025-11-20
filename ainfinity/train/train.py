import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ainfinity.config import settings


def load_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.name_or_path,
        trust_remote_code=model_args.get("trust_remote_code", False),
        use_fast=model_args.use_fast,
        return_tensors="pt",
        revision=model_args.get("revision", None),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_args):
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.revision,
        token=settings.HF_TOKEN,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        token=settings.HF_TOKEN,
        config=config,
        revision=model_args.revision,
        torch_dtype=torch.bfloat16,
        attn_implementation=model_args.attn_implementation,
    )

    return model


@hydra.main(config_path="config", config_name="finetuning", version_base=None)
def main(config: DictConfig):
    model_args = config.model
    dataset_args = config.dataset
    training_args = config.training_arguments

    print("model_args: ", model_args)
    print("dataset_args: ", dataset_args)
    print("training_args: ", training_args)


if __name__ == "__main__":
    main()
