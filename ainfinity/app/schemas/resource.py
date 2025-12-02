"""Resource configuration schemas"""

from typing import Optional

from pydantic import BaseModel, Field

from .base import InfraProvider


class ResourceConfig(BaseModel):
    """SkyPilot resource configuration"""

    infra: InfraProvider = Field(
        default=InfraProvider.VAST, description="Infrastructure provider"
    )
    accelerators: str = Field(
        default="RTX3090:1", description="GPU type and count (e.g., RTX3090:1, A100:4)"
    )
    disk_size: int = Field(default=100, ge=10, description="Disk size in GB")
    image_id: str = Field(
        default="docker:nvidia/cuda:12.1.1-devel-ubuntu20.04",
        description="Docker image to use",
    )
    memory: Optional[str] = Field(
        default="16+", description="Memory requirement (e.g., 32+)"
    )
    cpus: Optional[str] = Field(default="4+", description="CPU requirement (e.g., 8+)")
