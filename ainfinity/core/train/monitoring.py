import logging
import os
from typing import Any, Dict, Optional

import wandb

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Standardized tracker for SLM fine-tuning experiments.
    Wraps WandB and local logging to ensure consistent tracking.
    """

    def __init__(self, project_name: str, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self._setup()

    def _setup(self):
        """Initialize tracking backends."""
        logger.info(f"Initializing experiment: {self.project_name} / {self.run_name}")

        # Initialize WandB
        try:
            wandb.init(project=self.project_name, name=self.run_name, config=self.config)
            logger.info("WandB initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            # Continue even if WandB fails, relying on local logs

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all backends."""
        logger.info(f"Step {step}: {metrics}")
        if wandb.run:  # type: ignore
            wandb.log(metrics, step=step)  # type: ignore

    def log_artifact(self, artifact_path: str, artifact_type: str):
        """Log an artifact (model, dataset, etc.)."""
        logger.info(f"Logging artifact: {artifact_path} ({artifact_type})")
        if wandb.run:  # type: ignore
            artifact = wandb.Artifact(name=os.path.basename(artifact_path), type=artifact_type)  # type: ignore
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)  # type: ignore

    def finish(self):
        """Clean up and finish the experiment."""
        logger.info("Finishing experiment.")
        if wandb.run:
            wandb.finish()


def get_tracker(project_name: str, run_name: str, config: Dict[str, Any]) -> ExperimentTracker:
    return ExperimentTracker(project_name, run_name, config)
