from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class FineTuneRequest(BaseModel):
    model_name: str = "unsloth/llama-3-8b-bnb-4bit"
    data: List[Dict[str, Any]]  # Array of JSONL records
    params: Optional[Dict[str, Any]] = {}

class DeployRequest(BaseModel):
    model_id: str

class ModelInfo(BaseModel):
    model_id: str
    status: str
    base_model: str
    created_at: str
    artifact_path: Optional[str] = None
