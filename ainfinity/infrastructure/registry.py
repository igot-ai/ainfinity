import json
import os
from datetime import datetime
from typing import List, Optional

from train.schemas import ModelInfo

REGISTRY_FILE = "train/services/registry.json"


class ModelRegistry:
    def __init__(self):
        self._load_registry()

    def _load_registry(self):
        if not os.path.exists(REGISTRY_FILE):
            self.models = {}
            self._save_registry()
        else:
            with open(REGISTRY_FILE, "r") as f:
                self.models = json.load(f)

    def _save_registry(self):
        """
        This is an example of a registry from File
        @TODO: integrate model registry with Datahub
        https://opendatahub.io/docs/working-with-model-registries/#creating-a-model-registry_model-registry
        """
        with open(REGISTRY_FILE, "w") as f:
            json.dump(self.models, f, indent=2)

    def register_model(
        self, model_id: str, base_model: str, status: str = "training"
    ) -> ModelInfo:
        model_info = {
            "model_id": model_id,
            "status": status,
            "base_model": base_model,
            "created_at": datetime.now().isoformat(),
            "artifact_path": None,
        }
        self.models[model_id] = model_info
        self._save_registry()
        return ModelInfo(**model_info)

    def update_status(
        self, model_id: str, status: str, artifact_path: Optional[str] = None
    ):
        if model_id in self.models:
            self.models[model_id]["status"] = status
            if artifact_path:
                self.models[model_id]["artifact_path"] = artifact_path
            self._save_registry()

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        data = self.models.get(model_id)
        return ModelInfo(**data) if data else None

    def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(**m) for m in self.models.values()]


registry = ModelRegistry()
