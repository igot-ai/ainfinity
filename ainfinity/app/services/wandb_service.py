from typing import Any, Dict, List, Optional

import pandas as pd
from wandb import Api

from ainfinity.utils import logger, settings


class WandbService:
    """Service for interacting with Weights & Biases (WandB) API."""

    def __init__(self):
        """Initialize WandB service with API credentials."""
        self.wan = Api(api_key=settings.WANDB_API_KEY)

    def get_project_info(self):
        """Get the first accessible project's entity and name.

        Returns:
            Tuple of (entity, project_name) or empty list if error occurs
        """
        try:
            return self.wan.projects()[0].entity, self.wan.projects()[0].name
        except Exception as e:
            logger.error(f"Error getting projects: {e}")
            return []

    def list_all_paths(self, max_paths: int = 10) -> List[str]:
        """
        List ALL entities, projects, and run IDs accessible to you.

        Args:
            max_paths: Maximum number of paths to list (default: 10)

        Returns:
            List of paths
        """
        try:
            res = []
            projects = self.wan.projects()

            for project in projects:
                entity = project.entity
                project_name = project.name

                logger.info(f"Fetching runs from {entity}/{project_name}...")

                runs = self.get_project_runs(entity, project_name)
                res.extend([f"{entity}/{project_name}/{run.id}" for run in runs])
                if len(res) > max_paths:
                    return res[:max_paths]
            logger.info(f"Found {len(res)} paths")
            return res
        except Exception as e:
            logger.error(f"Error listing all paths: {e}")
            return []

    def get_project_runs(
        self, entity: str, project: str, filters: Optional[Dict[str, Any]] = None, order: str = "-created_at"
    ):
        """
        Get all runs from a specific project.

        Args:
            entity: The entity/username
            project: The project name
            filters: Optional filters (e.g., {"state": "finished", "tags": {"$in": ["training"]}})
            order: Sort order (default: newest first)

        Returns:
            List of runs
        """
        try:
            path = f"{entity}/{project}"
            runs = self.wan.runs(path, filters=filters, order=order)
            return list(runs)
        except Exception as e:
            logger.error(f"Error getting runs from {entity}/{project}: {e}")
            return []

    def get_run_by_id(self, entity: str, project: str, run_id: str):
        """Get a specific WandB run by its ID.

        Args:
            entity: The WandB entity/username
            project: The project name
            run_id: The unique run identifier

        Returns:
            Run object or None if not found
        """
        try:
            path = f"{entity}/{project}/{run_id}"
            return self.wan.run(path)
        except Exception as e:
            logger.error(f"Error getting run {run_id}: {e}")
            return None

    def get_run_metrics(self, entity: str, project: str, run_id: str, keys: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get training metrics/history from a specific run.

        Args:
            entity: The entity/username
            project: The project name
            run_id: The run ID
            keys: Specific metric keys to retrieve (e.g., ["loss", "grad_norm"])
                 If None, retrieves all metrics

        Returns:
            DataFrame with metrics history
        """
        try:
            run = self.get_run_by_id(entity, project, run_id)
            if not run:
                return pd.DataFrame()

            history = run.history(keys=keys) if keys else run.history()
            return history
        except Exception as e:
            logger.error(f"Error getting metrics for run {run_id}: {e}")
            return pd.DataFrame()

    def get_training_metrics_summary(self, entity: str, project: str, run_id: str) -> Dict[str, Any]:
        """Get comprehensive training metrics summary for a specific run.

        Args:
            entity: The WandB entity/username
            project: The project name
            run_id: The unique run identifier

        Returns:
            Dictionary with training metrics including loss, grad_norm, learning_rate, state, etc.
        """
        try:
            run = self.get_run_by_id(entity, project, run_id)
            if not run:
                print(f"Run {run_id} not found")
                return {}

            summary = dict(run.summary)
            config = dict(run.config)

            # Extract common training metrics
            training_summary = {
                "run_id": run_id,
                "run_name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "runtime": run.summary.get("_runtime", 0),
                "config": config,
                "metrics": {
                    "train_loss": summary.get("train/loss", summary.get("loss")),
                    "eval_loss": summary.get("eval/loss", summary.get("eval_loss")),
                    "grad_norm": summary.get("grad_norm"),
                    "learning_rate": summary.get("learning_rate", summary.get("lr")),
                    "epoch": summary.get("epoch"),
                    "step": summary.get("_step"),
                },
                "all_summary_metrics": summary,
            }

            return training_summary
        except Exception as e:
            logger.error(f"Error getting training summary for run {run_id}: {e}")
            return {}

    def get_training_history(
        self,
        entity: str,
        project: str,
        run_id: str,
        include_system_metrics: bool = False,
    ) -> pd.DataFrame:
        """
        Get comprehensive training history including loss, gradient norm, learning rate, etc.

        Args:
            entity: The entity/username
            project: The project name
            run_id: The run ID
            include_system_metrics: Whether to include system metrics like GPU, memory usage

        Returns:
            DataFrame with all training metrics over time
        """
        try:
            # Common training metrics
            metric_keys = [
                "_step",
                "_timestamp",
                "epoch",
                # Loss metrics
                "train/loss",
                "loss",
                "train_loss",
                "eval/loss",
                "eval_loss",
                "validation_loss",
                # Gradient metrics
                "grad_norm",
                "gradient_norm",
                "train/grad_norm",
                # Learning rate
                "learning_rate",
                "lr",
                "train/learning_rate",
                # Accuracy metrics (if available)
                "train/accuracy",
                "accuracy",
                "eval/accuracy",
                "eval_accuracy",
                # Perplexity (for language models)
                "train/perplexity",
                "perplexity",
                "eval/perplexity",
                "eval_perplexity",
            ]

            # Add system metrics if requested
            if include_system_metrics:
                metric_keys.extend(
                    [
                        "system/gpu.0.memory",
                        "system/gpu.0.gpu",
                        "system/memory",
                        "system/cpu",
                    ]
                )

            history = self.get_run_metrics(entity, project, run_id, keys=metric_keys)

            if not history.empty:
                # Remove columns that are all NaN
                history = history.dropna(axis=1, how="all")

                # Rename columns to standard names for consistency
                column_mapping = {
                    "loss": "train/loss",
                    "train_loss": "train/loss",
                    "eval_loss": "eval/loss",
                    "validation_loss": "eval/loss",
                    "gradient_norm": "grad_norm",
                    "train/gradient_norm": "grad_norm",
                    "train/grad_norm": "grad_norm",
                    "lr": "learning_rate",
                    "train/learning_rate": "learning_rate",
                    "accuracy": "train/accuracy",
                    "eval_accuracy": "eval/accuracy",
                    "perplexity": "train/perplexity",
                    "eval_perplexity": "eval/perplexity",
                }

                # Only rename columns that exist
                existing_mappings = {k: v for k, v in column_mapping.items() if k in history.columns}
                history = history.rename(columns=existing_mappings)

                # Remove duplicate columns (keep the first one)
                history = history.loc[:, ~history.columns.duplicated()]

            return history
        except Exception as e:
            logger.error(f"Error getting training history for run {run_id}: {e}")
            return pd.DataFrame()

    def get_reports(self, path: str):
        """Get all reports for a specific project.

        Args:
            path: Project path in format 'entity/project'

        Returns:
            List of report objects or empty list if error occurs
        """
        try:
            return self.wan.reports(path=path)
        except Exception as e:
            logger.error(f"Error getting reports for {path}: {e}")
            return []

    def sync_tensorboard(
        self,
        root_dir: str,
        run_id: Optional[str] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
    ):
        """Sync TensorBoard logs to WandB for visualization.

        Args:
            root_dir: Directory containing TensorBoard logs
            run_id: Optional run ID to associate logs with
            project: Optional project name
            entity: Optional entity/username
        """
        try:
            self.wan.sync_tensorboard(root_dir, run_id=run_id, project=project, entity=entity)
            logger.info(f"Successfully synced TensorBoard logs from {root_dir}")
        except Exception as e:
            logger.error(f"Error syncing TensorBoard logs: {e}")


wan = WandbService()
