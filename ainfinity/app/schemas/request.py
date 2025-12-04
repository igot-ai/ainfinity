"""API request schemas"""

from pydantic import BaseModel, Field

from .resource import ResourceConfig
from .training import Datalog, ModelInfo, TrainingParams


class TrainingJobRequest(BaseModel):
    """Request to create and launch a training job"""

    job_name: str = Field(..., description="Unique name for the job", min_length=1, max_length=64)
    resource: ResourceConfig = Field(default_factory=ResourceConfig, description="Resource configuration")
    model: ModelInfo = Field(..., description="Model configuration")
    dataset: Datalog = Field(..., description="Dataset configuration")
    training_params: TrainingParams = Field(default_factory=TrainingParams, description="Training arguments")
    detach: bool = Field(default=True, description="Run job in detached mode")
    down: bool = Field(default=True, description="Terminate cluster after job completion")
