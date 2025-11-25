import os
import subprocess  # nosec B404
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from .schemas import DeployRequest, ModelInfo
from .infrastructure import registry, sky_service

app = FastAPI(title="SLM Fine-tuning Microservice")

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def run_sky_command(cmd: str):
    """Helper to run sky command in background."""
    # In production, use Celery or similar
    print(f"Executing: {cmd}")
    subprocess.Popen(cmd, shell=True)  # nosec B602


@app.post("/fine-tune", response_model=ModelInfo)
async def fine_tune(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    model_name: str = Form("unsloth/Llama-3.2-1B-bnb-4bit"),
    epochs: int = Form(1),
):
    model_id = str(uuid.uuid4())[:8]

    # Save and merge uploaded files
    dataset_path = os.path.join(DATA_DIR, f"{model_id}.jsonl")

    with open(dataset_path, "wb") as outfile:
        for file in files:
            content = await file.read()
            # Ensure newline between files if missing
            if content and not content.endswith(b"\n"):
                content += b"\n"
            outfile.write(content)

    # Register model
    # model_info = registry.register_model(model_id, model_name)

    # Launch training
    cmd = sky_service.launch_training(
        model_id, model_name, os.path.abspath(dataset_path)
    )
    background_tasks.add_task(run_sky_command, cmd)

    return model_info


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    return registry.list_models()


@app.post("/deploy")
async def deploy_model(request: DeployRequest, background_tasks: BackgroundTasks):
    model = registry.get_model(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_path = model.artifact_path

    cmd, port = sky_service.launch_serving(request.model_id, model_path)
    background_tasks.add_task(run_sky_command, cmd)

    return {
        "status": "deployment_initiated",
        "job_id": f"serve-{request.model_id}",
        "port": port,
    }
