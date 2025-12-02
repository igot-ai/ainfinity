"""API response schemas"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .base import JobStatus
from .metrics import EvaluationMetrics, GPUMetrics, TrainingMetrics
from .request import TrainingRequest
from .resource import ResourceConfig


class JobInfo(BaseModel):
    """Information about a training job"""

    job_name: str
    cluster_name: str
    status: JobStatus
    resources: ResourceConfig
    training: TrainingRequest
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    error_message: Optional[str] = None

    # Metrics
    training_metrics: Optional[TrainingMetrics] = Field(
        default=None, description="Current training metrics"
    )
    evaluation_metrics: Optional[EvaluationMetrics] = Field(
        default=None, description="Latest evaluation results"
    )
    gpu_metrics: Optional[GPUMetrics] = Field(
        default=None, description="GPU utilization metrics"
    )


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
