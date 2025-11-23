import uuid
import json
import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from schemas import FineTuneRequest, DeployRequest, ModelInfo
from services.registry import registry
from services.sky_wrapper import sky_service
import subprocess

app = FastAPI(title="SLM Fine-tuning Microservice")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def run_sky_command(cmd: str):
    """Helper to run sky command in background."""
    # In production, use Celery or similar
    print(f"Executing: {cmd}")
    subprocess.Popen(cmd, shell=True)

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from typing import List, Optional
import shutil

# ... imports ...

@app.post("/fine-tune", response_model=ModelInfo)
async def fine_tune(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model_name: str = Form("unsloth/llama-3-8b-bnb-4bit"),
    epochs: int = Form(1)
):
    model_id = str(uuid.uuid4())[:8]
    
    # Save and merge uploaded files
    dataset_path = os.path.join(DATA_DIR, f"{model_id}.jsonl")
    
    with open(dataset_path, 'wb') as outfile:
        for file in files:
            content = await file.read()
            # Ensure newline between files if missing
            if content and not content.endswith(b'\n'):
                content += b'\n'
            outfile.write(content)
    
    # Register model
    model_info = registry.register_model(model_id, model_name)
    
    # Launch training
    cmd = sky_service.launch_training(model_id, model_name, os.path.abspath(dataset_path))
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
    
    # Assuming artifact path is set after training
    # For demo, we'll use the base model if artifact path is missing
    model_path = model.artifact_path or model.base_model
    
    cmd = sky_service.launch_serving(request.model_id, model_path)
    background_tasks.add_task(run_sky_command, cmd)
    
    return {"status": "deployment_initiated", "job_id": f"serve-{request.model_id}"}
