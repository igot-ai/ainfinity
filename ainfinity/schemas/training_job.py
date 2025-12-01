from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a training job"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class InfraProvider(str, Enum):
    """Infrastructure provider options"""

    VAST = "vast"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class ResourceConfig(BaseModel):
    """SkyPilot resource configuration"""

    infra: InfraProvider = Field(default=InfraProvider.VAST, description="Infrastructure provider")
    accelerators: str = Field(default="RTX3090:1", description="GPU type and count (e.g., RTX3090:1, A100:4)")
    disk_size: int = Field(default=100, ge=10, description="Disk size in GB")
    image_id: str = Field(default="docker:nvidia/cuda:12.1.1-devel-ubuntu20.04", description="Docker image to use")
    memory: Optional[str] = Field(default=None, description="Memory requirement (e.g., 32+)")
    cpus: Optional[str] = Field(default=None, description="CPU requirement (e.g., 8+)")


class TrainingConfig(BaseModel):
    """Training configuration"""

    config_file: str = Field(default="finetuning", description="Hydra config name (without .yaml)")
    dataset: Optional[str] = Field(default=None, description="Override dataset config")
    model: Optional[str] = Field(default=None, description="Override model config")
    num_train_epochs: Optional[int] = Field(default=None, ge=1, description="Number of training epochs")
    per_device_train_batch_size: Optional[int] = Field(default=None, ge=1, description="Training batch size per device")
    learning_rate: Optional[float] = Field(default=None, gt=0, description="Learning rate")
    output_dir: Optional[str] = Field(default=None, description="Output directory for checkpoints")
    accelerate_config: str = Field(default="ds2_config.yaml", description="Accelerate config file name")
    extra_args: Optional[Dict[str, str]] = Field(default=None, description="Additional Hydra overrides")


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


class LaunchJobRequest(BaseModel):
    """Request to launch a training job"""

    job_name: str = Field(..., description="Unique name for the job", min_length=1, max_length=64)
    resources: ResourceConfig = Field(default_factory=ResourceConfig, description="Resource configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    detach: bool = Field(default=True, description="Run job in detached mode")
    down: bool = Field(default=True, description="Terminate cluster after job completion")


class JobInfo(BaseModel):
    """Information about a training job"""

    job_name: str
    cluster_name: str
    status: JobStatus
    resources: ResourceConfig
    training: TrainingConfig
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Metrics
    training_metrics: Optional[TrainingMetrics] = Field(default=None, description="Current training metrics")
    evaluation_metrics: Optional[EvaluationMetrics] = Field(default=None, description="Latest evaluation results")
    gpu_metrics: Optional[GPUMetrics] = Field(default=None, description="GPU utilization metrics")


class JobResponse(BaseModel):
    """Response for job operations"""

    success: bool
    message: str
    job_info: Optional[JobInfo] = None


class JobListResponse(BaseModel):
    """Response for listing jobs"""

    success: bool
    jobs: list[JobInfo]
    total: int
