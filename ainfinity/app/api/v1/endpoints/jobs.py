"""
Jobs Router - Training job management endpoints
"""

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from ainfinity.app.api.dependencies import get_training_service
from ainfinity.app.schemas import JobListResponse, JobResponse, LaunchJobRequest

router = APIRouter()


@router.post("", response_model=JobResponse, status_code=201)
async def launch_job(request: LaunchJobRequest):
    """
    Launch a new training job

    Args:
        request: Job configuration

    Returns:
        Job information
    """
    service = get_training_service()
    job_info = service.launch_job(request)
    return JobResponse(
        success=True,
        message=f"Job '{request.job_name}' launched successfully",
        job_info=job_info,
    )


@router.get("", response_model=JobListResponse)
async def list_jobs():
    """
    List all training jobs

    Returns:
        List of all jobs
    """
    service = get_training_service()
    jobs = service.list_jobs()
    return JobListResponse(success=True, jobs=jobs, total=len(jobs))


@router.get("/{job_name}", response_model=JobResponse)
async def get_job(job_name: str):
    """
    Get status of a specific job

    Args:
        job_name: Name of the job

    Returns:
        Job information
    """
    service = get_training_service()
    job_info = service.get_job_status(job_name)
    return JobResponse(success=True, message=f"Job '{job_name}' status retrieved", job_info=job_info)


@router.delete("/{job_name}", response_model=JobResponse)
async def stop_job(job_name: str):
    """
    Stop a running job

    Args:
        job_name: Name of the job

    Returns:
        Updated job information
    """
    service = get_training_service()
    job_info = service.stop_job(job_name)
    return JobResponse(
        success=True,
        message=f"Job '{job_name}' stopped successfully",
        job_info=job_info,
    )


@router.get("/{job_name}/logs")
async def get_job_logs(
    job_name: str,
    tail: int = Query(100, ge=1, le=10000, description="Number of lines to retrieve"),
):
    """
    Get logs from a job

    Args:
        job_name: Name of the job
        tail: Number of lines to retrieve (default: 100, max: 10000)

    Returns:
        Job logs
    """
    service = get_training_service()
    logs = service.get_job_logs(job_name, tail=tail)
    return JSONResponse(content={"success": True, "job_name": job_name, "logs": logs, "lines": tail})


@router.delete("/{job_name}/delete")
async def delete_job(job_name: str):
    """
    Delete a job from database (does not stop the cluster)

    Args:
        job_name: Name of the job

    Returns:
        Success message
    """
    service = get_training_service()
    service.delete_job(job_name)
    return JSONResponse(content={"success": True, "message": f"Job '{job_name}' deleted from database"})


@router.get("/{job_name}/metrics")
async def get_job_metrics(job_name: str):
    """
    Get detailed metrics for a training job

    Args:
        job_name: Name of the job

    Returns:
        Training, evaluation and GPU metrics
    """
    service = get_training_service()
    job_info = service.get_job_metrics(job_name)

    return JSONResponse(
        content={
            "success": True,
            "job_name": job_name,
            "status": job_info.status,
            "training_metrics": (job_info.training_metrics.model_dump() if job_info.training_metrics else None),
            "evaluation_metrics": (job_info.evaluation_metrics.model_dump() if job_info.evaluation_metrics else None),
            "gpu_metrics": (job_info.gpu_metrics.model_dump() if job_info.gpu_metrics else None),
        }
    )
