"""Training configuration schemas for model, dataset, and training parameters"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .base import (
    AttnImpl,
    DatalogSchema,
    DataSource,
    EvalStrategy,
    PaddingStrategy,
    SaveStrategy,
    SchedulerType,
)


class ModelInfo(BaseModel):
    """Model configuration for training"""

    pretrained_model_name_or_path: str = Field(..., description="Pretrained model name or path")
    model_max_length: int = Field(default=2048, ge=512, description="Maximum sequence length for the model")
    attn_implementation: AttnImpl = Field(default=AttnImpl.SPDA, description="Attention implementation to use")
    trust_remote_code: bool = Field(default=True, description="Trust remote code when loading model")
    use_fast: bool = Field(default=True, description="Use fast tokenizer")
    revision: str = Field(default="main", description="Model revision/branch")
    padding: PaddingStrategy = Field(default=PaddingStrategy.LONGEST, description="Padding strategy for tokenizer")
    padding_side: str = Field(default="left", description="Padding side for tokenizer")
    truncation_side: str = Field(default="right", description="Truncation side for tokenizer")
    instruction_template: str = Field(default="<|im_start|>user", description="Instruction template for completion")
    response_template: str = Field(default="<|im_start|>assistant", description="Response template for completion")


class DatasetSplit(BaseModel):
    """Configuration for a single dataset split"""

    name: str = Field(..., description="Name of the split (e.g., 'train', 'validation')")
    max_samples: Optional[int] = Field(default=None, description="Maximum number of samples to use from this split")


class DatasetInfo(BaseModel):
    """Dataset configuration for training"""

    name: str = Field(..., description="Name or path of the dataset")
    text_col: str = Field(..., description="Column name containing text data")
    split: Dict[str, DatasetSplit] = Field(..., description="Dataset splits configuration (train, validation, etc.)")
    revision: Optional[str] = Field(default="main", description="Dataset revision/branch to use")
    num_proc: Optional[int] = Field(default=1, ge=1, description="Number of processes for dataset loading")


class Datalog(BaseModel):
    source: DataSource = Field(default=DataSource.DATALOG, description="Data source type")
    schema_type: DatalogSchema = Field(..., description="Type of datalog schema")
    urls: List[str] = Field(..., description="List of URLs to the datalog files")
    shuffle: bool = Field(default=True, description="Whether to shuffle the dataset")
    train_eval_split: float = Field(default=0.8, ge=0.0, le=1.0, description="Train-test split ratio")
    train_max_samples: Optional[int] = Field(default=None, description="Maximum samples for training set")
    validation_max_samples: Optional[int] = Field(default=None, description="Maximum samples for validation set")


class TrainingParams(BaseModel):
    """Training arguments configuration"""

    output_dir: str = Field(default="./outputs/finetuning", description="Output directory for checkpoints")
    num_train_epochs: int = Field(default=1, ge=1, description="Number of training epochs")
    per_device_train_batch_size: int = Field(default=4, ge=1, description="Training batch size per device")
    per_device_eval_batch_size: int = Field(default=4, ge=1, description="Evaluation batch size per device")
    eval_strategy: EvalStrategy = Field(default=EvalStrategy.EPOCH, description="Evaluation strategy during training")
    save_strategy: SaveStrategy = Field(default=SaveStrategy.EPOCH, description="Model checkpoint saving strategy")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Number of gradient accumulation steps")
    bf16: bool = Field(default=True, description="Use bfloat16 precision")
    fp16: bool = Field(default=False, description="Use fp16 precision")
    logging_steps: int = Field(default=10, ge=1, description="Log every X steps")
    save_steps: int = Field(default=500, ge=1, description="Save checkpoint every X steps")
    eval_steps: int = Field(default=500, ge=1, description="Evaluate every X steps")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    lr_scheduler_type: SchedulerType = Field(
        default=SchedulerType.LINEAR,
        description="Type of LR scheduler (e.g., linear, cosine)",
    )
    warmup_steps: int = Field(default=0, ge=0, description="Number of warmup steps for LR scheduler")
    warmup_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Warmup ratio for LR scheduler")
    learning_rate: float = Field(default=2e-5, gt=0, description="Initial learning rate")
