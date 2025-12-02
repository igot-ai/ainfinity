"""API request schemas"""

from pydantic import BaseModel, Field

from .resource import ResourceConfig
from .training import DatasetInfo, ModelInfo, TrainingParams


class TrainingRequest(BaseModel):
    """Training configuration for API requests"""

    model: ModelInfo = Field(..., description="Model configuration")
    dataset: DatasetInfo = Field(..., description="Dataset configuration")
    training_args: TrainingParams = Field(
        default_factory=TrainingParams, description="Training arguments"
    )


class LaunchJobRequest(BaseModel):
    """Request to launch a training job"""

    job_name: str = Field(
        ..., description="Unique name for the job", min_length=1, max_length=64
    )
    resources: ResourceConfig = Field(
        default_factory=ResourceConfig, description="Resource configuration"
    )
    training: TrainingRequest = Field(
        default_factory=TrainingRequest, description="Training configuration"
    )
    detach: bool = Field(default=True, description="Run job in detached mode")
    down: bool = Field(
        default=True, description="Terminate cluster after job completion"
    )
