import os
import subprocess
import uuid
import json
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any

TEMPLATE_DIR = "templates"
GENERATED_DIR = "generated_configs"

class SkyWrapper:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        os.makedirs(GENERATED_DIR, exist_ok=True)

    def _generate_config(self, template_name: str, context: Dict[str, Any]) -> str:
        template = self.env.get_template(template_name)
        content = template.render(context)
        
        filename = f"{context['job_name']}.yaml"
        filepath = os.path.join(GENERATED_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath

    def launch_training(self, model_id: str, base_model: str, dataset_path: str) -> str:
        job_name = f"train-{model_id}"
        config_path = self._generate_config("train_sky.j2", {
            "job_name": job_name,
            "accelerator_type": "L4:1", # Default to L4
            "base_model": base_model,
            "dataset_path": dataset_path,
            "output_dir": f"/artifacts/{model_id}", # Placeholder
            "project_name": "slm-microservice",
            "run_name": model_id
        })
        
        # In a real app, we'd use the python API: sky.launch(...)
        # But for now, we'll shell out or just return the command
        cmd = f"sky launch -c {job_name} {config_path} -d --env WANDB_API_KEY=$WANDB_API_KEY"
        return cmd

    def launch_serving(self, model_id: str, model_path: str) -> str:
        job_name = f"serve-{model_id}"
        config_path = self._generate_config("serve_sky.j2", {
            "job_name": job_name,
            "accelerator_type": "L4:1",
            "model_path": model_path
        })
        
        cmd = f"sky launch -c {job_name} {config_path} -d"
        return cmd

sky_service = SkyWrapper()
