"""Metrics schemas for training, evaluation, and GPU monitoring"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TrainingMetrics(BaseModel):
    """Training metrics during fine-tuning"""

    current_epoch: Optional[int] = Field(default=None, description="Current training epoch")
    total_epochs: Optional[int] = Field(default=None, description="Total number of epochs")
    current_step: Optional[int] = Field(default=None, description="Current training step")
    total_steps: Optional[int] = Field(default=None, description="Total training steps")
    loss: Optional[float] = Field(default=None, description="Current training loss")
    learning_rate: Optional[float] = Field(default=None, description="Current learning rate")
    grad_norm: Optional[float] = Field(default=None, description="Gradient norm")
    samples_per_second: Optional[float] = Field(default=None, description="Training throughput")
    last_updated: Optional[datetime] = Field(default=None, description="Last metrics update time")


class EvaluationMetrics(BaseModel):
    """Evaluation results"""

    eval_loss: Optional[float] = Field(default=None, description="Evaluation loss")
    eval_accuracy: Optional[float] = Field(default=None, description="Evaluation accuracy")
    eval_perplexity: Optional[float] = Field(default=None, description="Perplexity score")
    eval_samples: Optional[int] = Field(default=None, description="Number of evaluation samples")
    last_evaluated: Optional[datetime] = Field(default=None, description="Last evaluation time")


class GPUMetrics(BaseModel):
    """GPU utilization metrics"""

    gpu_utilization: Optional[float] = Field(default=None, ge=0, le=100, description="GPU utilization %")
    gpu_memory_used: Optional[float] = Field(default=None, description="GPU memory used (GB)")
    gpu_memory_total: Optional[float] = Field(default=None, description="Total GPU memory (GB)")
    gpu_temperature: Optional[float] = Field(default=None, description="GPU temperature (°C)")
    power_usage: Optional[float] = Field(default=None, description="Power usage (W)")
    last_updated: Optional[datetime] = Field(default=None, description="Last metrics update time")
