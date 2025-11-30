from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel, TrainingArguments

from ainfinity.utils import settings


@dataclass
class ComputeSettings:
    dtype: torch.dtype
    attn_impl: str
    reason: str


def print_trainable_parameters(model: PreTrainedModel):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    summary = f"Trainable: {trainable:,} | Total: {total:,} | {100 * trainable / total:.2f}% trainable"
    return summary


def select_torch_compute(
    fp16: Optional[bool] = False, bf16: Optional[bool] = False, attn_impl: Optional[str] = None
) -> ComputeSettings:
    """
    Select compute dtype and attention implementation based on GPU and user preference.

    Args:
        fp16 (bool): If True, force float16.
        bf16 (bool): If True, force bfloat16 (only if GPU supports it).
        attn_impl

    Returns:
        ComputeSettings: dtype, attention implementation, and reason
    """
    if not torch.cuda.is_available():
        # CPU fallback
        return ComputeSettings(dtype=torch.float32, attn_impl="eager", reason="CUDA not available")

    major, minor = torch.cuda.get_device_capability()

    # Priority: bf16 > fp16 > automatic
    if bf16 and major >= 8:
        if attn_impl is None:
            attn_impl = "flash_attention_2"
        return ComputeSettings(
            dtype=torch.bfloat16,
            attn_impl=attn_impl,
            reason=f"Forced BF16 and GPU {major}.{minor} supports it",
        )

    if fp16:
        return ComputeSettings(dtype=torch.float16, attn_impl="eager", reason="Forced FP16")

    # Automatic selection if no forced dtype
    if major >= 8:
        if attn_impl is None:
            attn_impl = "flash_attention_2"
        return ComputeSettings(
            dtype=torch.bfloat16,
            attn_impl=attn_impl,
            reason=f"GPU {major}.{minor} supports BF16 & Flash Attention 2",
        )

    return ComputeSettings(
        dtype=torch.float16,
        attn_impl="eager",
        reason=f"GPU {major}.{minor} does not fully support BF16 or Flash Attention 2",
    )


def configure_hub_settings(training_args: TrainingArguments):
    """
        Validate and configure Hugging Face Hub settings for training.

        This function ensures that when `push_to_hub=True`, the necessary
        Hub-related arguments (`hub_model_id`, `hub_token`) are provided. If
        they are missing, the function attempts to populate them from the global
        `settings`. If required fields are still unavailable, a ValueError is raised.

        Parameters
        ----------
        training_args : TrainingArguments
            The HuggingFace TrainingArguments object containing Hub configuration.
    configure_hub_settings
        Returns
        -------
        TrainingArguments
            The updated `training_args` with validated or auto-filled Hub settings.

        Raises
        ------
        ValueError
            If `push_to_hub=True` but neither `training_args.hub_model_id`
            nor `settings.HUB_MODEL_ID` is provided.
    """
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            if settings.HUB_MODEL_ID is None:
                raise ValueError("hub_model_id is required when push_to_hub=True.")

            training_args.hub_model_id = settings.HUB_MODEL_ID
        if training_args.hub_token is None:
            training_args.hub_token = settings.HF_TOKEN

    return training_args
