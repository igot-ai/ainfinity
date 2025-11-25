import os
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader

# Because the mounted folder would be set to /train
TEMPLATE_DIR = "/train/templates"
GENERATED_DIR = "/train/generated_configs"


class SkyWrapper:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))  # nosec B701
        os.makedirs(GENERATED_DIR, exist_ok=True)

    def _generate_config(self, template_name: str, context: Dict[str, Any]) -> str:
        template = self.env.get_template(template_name)
        content = template.render(context)

        filename = f"{context['model_id']}.yaml"
        filepath = os.path.join(GENERATED_DIR, filename)

        with open(filepath, "w") as f:
            f.write(content)

        return filepath

    def launch_training(self, model_id: str, base_model: str, dataset_path: str) -> str:
        job_name = f"train-{model_id}"
        config_path = self._generate_config(
            "train_sky.j2",
            {
                "model_id": model_id,
                "accelerator_type": "L4:1",  # Default to L4
                "base_model": base_model,
                "dataset_path": dataset_path,
                "output_dir": f"/models/{model_id}",  # Placeholder
                "project_name": "slm-microservice",
                "run_name": model_id,
            },
        )

        # In a real app, we'd use the python API: sky.launch(...)
        # But for now, we'll shell out or just return the command
        cmd = f"sky launch -c {job_name} {config_path} -d --down --env WANDB_API_KEY=$WANDB_API_KEY -y"
        return cmd

    def launch_serving(self, model_id: str, model_path: str) -> str:
        job_name = f"serve-{model_id}"

        # Generate a unique port based on model_id (range: 8001-8999)
        # This ensures each model gets a consistent, unique port
        import hashlib

        port_hash = int(hashlib.md5(model_id.encode()).hexdigest()[:4], 16)
        port = 8001 + (port_hash % 999)

        config_path = self._generate_config(
            "serve_sky.j2",
            {
                "model_id": model_id,
                "accelerator_type": "L4:1",
                "model_path": model_path,
                "port": port,
            },
        )

        cmd = f"sky launch -c {job_name} {config_path} -d -y"
        return cmd, port


sky_service = SkyWrapper()
