import json
import os
import subprocess  # nosec B404
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore

from ainfinity.app.exceptions import JobAlreadyExistsException
from ainfinity.app.schemas import (
    EvaluationMetrics,
    GPUMetrics,
    JobInfo,
    JobStatus,
    LaunchJobRequest,
    TrainingMetrics,
)
from ainfinity.utils import load_json, save_json


# Custom YAML representer for literal block scalars
class literal_str(str):
    """String subclass for YAML literal block scalar (|)"""


def literal_str_representer(dumper, data):
    """Represent literal_str as YAML literal block scalar"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


# Register the custom representer
yaml.add_representer(literal_str, literal_str_representer)


class SkyPilotService:
    """Service to manage training jobs with SkyPilot"""

    def __init__(self, workspace_root: Optional[str] = None):
        """
        Initialize SkyPilot service

        Args:
            workspace_root: Root directory of the workspace. If None, uses current directory.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.template_path = (
            self.workspace_root / "ainfinity" / "skylaunch" / "deepspeed.yaml"
        )
        self.jobs_db_path = self.workspace_root / ".skypilot_jobs.json"
        self._ensure_jobs_db()

    def _ensure_jobs_db(self):
        """Ensure jobs database file exists"""
        if not self.jobs_db_path.exists():
            self._save_jobs_db({})

    def _load_jobs_db(self) -> Dict:
        """Load jobs database"""
        try:
            return load_json(self.jobs_db_path)
        except (FileNotFoundError, Exception):
            return {}

    def _save_jobs_db(self, jobs_db: Dict):
        """Save jobs database"""
        save_json(jobs_db, self.jobs_db_path)

    def _run_sky_command(
        self, cmd: List[str], check: bool = True, capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run a SkyPilot CLI command

        Args:
            cmd: Command as list of strings
            check: Whether to raise exception on non-zero exit code
            capture_output: Whether to capture output (False for interactive commands like launch)

        Returns:
            CompletedProcess object
        """
        print(f"Running SkyPilot command: {' '.join(cmd)}")
        print("-" * 80)

        try:
            if capture_output:
                # Capture output for parsing (used by status, logs, etc.)
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=check,
                    cwd=str(self.workspace_root),
                )  # nosec B603

                # Print captured output for visibility
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
            else:
                # Stream output in real-time (used by launch)
                result = subprocess.run(
                    cmd,
                    text=True,
                    check=check,
                    cwd=str(self.workspace_root),
                    stdout=None,  # Inherit parent's stdout
                    stderr=None,  # Inherit parent's stderr
                )  # nosec B603

            print("-" * 80)
            return result

        except subprocess.CalledProcessError as e:
            error_msg = f"SkyPilot command failed with exit code {e.returncode}"
            if hasattr(e, "stderr") and e.stderr:
                error_msg += f"\nError output: {e.stderr}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e

    def _generate_yaml_config(self, request: LaunchJobRequest) -> str:
        """
        Generate SkyPilot YAML configuration from request

        Args:
            request: Launch job request

        Returns:
            Path to generated YAML file
        """
        print("Request sent to create YAML config:", request)
        # Build the run command with training configuration
        run_commands = ["source .venv/bin/activate", ""]

        # Build accelerate launch command
        accelerate_config_path = f"./ainfinity/core/config/accelerate_config/{request.training.accelerate_config}"
        python_file = "./ainfinity/core/finetune.py"

        cmd_parts = [
            "accelerate launch",
            f"--config_file {accelerate_config_path}",
            python_file,
        ]

        # Add Hydra config overrides
        overrides = []
        if request.training.dataset:
            overrides.append(f"dataset={request.training.dataset}")
        if request.training.model:
            overrides.append(f"model={request.training.model}")
        if request.training.num_train_epochs:
            overrides.append(
                f"training_arguments.num_train_epochs={request.training.num_train_epochs}"
            )
        if request.training.per_device_train_batch_size:
            overrides.append(
                f"training_arguments.per_device_train_batch_size={request.training.per_device_train_batch_size}"
            )
        if request.training.learning_rate:
            overrides.append(
                f"training_arguments.learning_rate={request.training.learning_rate}"
            )
        if request.training.output_dir:
            overrides.append(
                f"training_arguments.output_dir={request.training.output_dir}"
            )

        # Temporary disabled extra arguments
        # # Add extra arguments
        # if request.training.extra_args:
        #     for key, value in request.training.extra_args.items():
        #         overrides.append(f"{key}={value}")

        if overrides:
            cmd_parts.extend(overrides)

        run_commands.append(" ".join(cmd_parts))

        # Create YAML configuration

        resources_config: Dict[str, Any] = {
            "infra": request.resources.infra.value,  # Convert enum to string
            "accelerators": request.resources.accelerators,
            "disk_size": request.resources.disk_size,
            "image_id": request.resources.image_id,
        }

        # Add optional resource fields
        if request.resources.memory:
            resources_config["memory"] = request.resources.memory
        if request.resources.cpus:
            resources_config["cpus"] = request.resources.cpus

        # Create setup and run scripts as literal block scalars (remove the " |" part!)
        setup_script = literal_str(
            "uv venv .venv --python=3.12\n"
            "source .venv/bin/activate\n"
            "uv sync --group gpu"
        )

        run_script = literal_str("\n".join(run_commands))

        config: Dict[str, Any] = {
            "name": request.job_name,
            "envs": {
                "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
                "HF_TOKEN": os.getenv("HF_TOKEN"),
            },
            "resources": resources_config,
            "workdir": ".",
            "setup": setup_script,
            "run": run_script,
        }

        yaml_dict = {
            k: v if not isinstance(v, literal_str) else str(v)
            for k, v in config.items()
        }
        print(f"YAML configuration: {json.dumps(yaml_dict, indent=4)}")

        # Write to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(self.workspace_root)
        )
        yaml.dump(
            config,
            temp_file,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        temp_file.close()

        return temp_file.name

    def launch_job(self, request: LaunchJobRequest) -> JobInfo:
        """
        Launch a training job with SkyPilot

        Args:
            request: Launch job request

        Returns:
            JobInfo object with job details
        """
        # Check if job name already exists
        jobs_db = self._load_jobs_db()
        if request.job_name in jobs_db:
            raise JobAlreadyExistsException(
                f"Job with name '{request.job_name}' already exists"
            )

        # Generate YAML configuration
        yaml_path = self._generate_yaml_config(request)

        try:
            # Build sky launch command
            cmd = ["sky", "launch", "-y"]

            if request.detach:
                cmd.append("-d")

            if request.down:
                cmd.append("--down")

            cmd.extend(["-c", request.job_name, yaml_path])

            # Launch the job (stream output in real-time)
            self._run_sky_command(cmd, capture_output=False)

            # Create job info
            job_info = JobInfo(
                job_name=request.job_name,
                cluster_name=request.job_name,
                status=JobStatus.PENDING,
                resources=request.resources,
                training=request.training,
                created_at=datetime.now(),
            )

            # Save to database
            jobs_db[request.job_name] = job_info.model_dump(mode="json")
            self._save_jobs_db(jobs_db)

            return job_info

        finally:
            # Clean up temporary YAML file
            try:
                os.unlink(yaml_path)
            except OSError:
                pass

    def get_job_status(self, job_name: str) -> JobInfo:
        """
        Get status of a training job

        Args:
            job_name: Name of the job

        Returns:
            JobInfo object with current status
        """
        jobs_db = self._load_jobs_db()

        if job_name not in jobs_db:
            raise ValueError(f"Job '{job_name}' not found")

        # Get stored job info
        job_data = jobs_db[job_name]

        # Query SkyPilot for current status
        try:
            result = self._run_sky_command(["sky", "status", job_name], check=False)

            # Parse output to determine status
            output = result.stdout.lower()

            if result.returncode != 0 or "not found" in output:
                job_data["status"] = JobStatus.UNKNOWN
            elif "init" in output or "pending" in output:
                job_data["status"] = JobStatus.PENDING
            elif "up" in output or "running" in output:
                job_data["status"] = JobStatus.RUNNING
                if not job_data.get("started_at"):
                    job_data["started_at"] = datetime.now().isoformat()
            elif "stopped" in output:
                job_data["status"] = JobStatus.STOPPED
                if not job_data.get("ended_at"):
                    job_data["ended_at"] = datetime.now().isoformat()

            # Update database
            jobs_db[job_name] = job_data
            self._save_jobs_db(jobs_db)

        except Exception as e:
            job_data["error_message"] = str(e)

        return JobInfo(**job_data)

    def stop_job(self, job_name: str) -> JobInfo:
        """
        Stop a running training job

        Args:
            job_name: Name of the job

        Returns:
            JobInfo object with updated status
        """
        jobs_db = self._load_jobs_db()

        if job_name not in jobs_db:
            raise ValueError(f"Job '{job_name}' not found")

        # Stop the cluster
        try:
            self._run_sky_command(["sky", "down", "-y", job_name], capture_output=False)

            # Update job info
            job_data = jobs_db[job_name]
            job_data["status"] = JobStatus.STOPPED
            job_data["ended_at"] = datetime.now().isoformat()

            jobs_db[job_name] = job_data
            self._save_jobs_db(jobs_db)

            return JobInfo(**job_data)

        except Exception as e:
            job_data = jobs_db[job_name]
            job_data["error_message"] = str(e)
            return JobInfo(**job_data)

    def list_jobs(self) -> List[JobInfo]:
        """
        List all training jobs

        Returns:
            List of JobInfo objects
        """
        jobs_db = self._load_jobs_db()

        jobs = []
        for job_name in jobs_db:
            try:
                job_info = self.get_job_status(job_name)
                jobs.append(job_info)
            except Exception:
                # If we can't get status, use stored data
                jobs.append(JobInfo(**jobs_db[job_name]))

        return jobs

    def get_job_logs(self, job_name: str, tail: int = 100) -> str:
        """
        Get logs from a training job

        Args:
            job_name: Name of the job
            tail: Number of lines to retrieve from end of log

        Returns:
            Log output as string
        """
        jobs_db = self._load_jobs_db()

        if job_name not in jobs_db:
            raise ValueError(f"Job '{job_name}' not found")

        try:
            # Get logs using sky logs command
            result = self._run_sky_command(
                ["sky", "logs", job_name, "--tail", str(tail)], check=False
            )
            return result.stdout
        except Exception as e:
            return f"Error retrieving logs: {str(e)}"

    def delete_job(self, job_name: str) -> bool:
        """
        Delete a job from the database

        Args:
            job_name: Name of the job

        Returns:
            True if deleted successfully
        """
        jobs_db = self._load_jobs_db()

        if job_name not in jobs_db:
            raise ValueError(f"Job '{job_name}' not found")

        # Remove from database
        del jobs_db[job_name]
        self._save_jobs_db(jobs_db)

        return True

    def _parse_training_metrics(self, logs: str) -> Optional[TrainingMetrics]:
        """
        Parse training metrics from logs

        Args:
            logs: Log output from training job

        Returns:
            TrainingMetrics object or None if not found
        """
        import re

        metrics = TrainingMetrics()

        try:
            # Parse epoch/step info: {'loss': 0.5, 'learning_rate': 2e-5, 'epoch': 1.5}
            epoch_match = re.search(r"'epoch':\s*([\d.]+)", logs)
            if epoch_match:
                metrics.current_epoch = int(float(epoch_match.group(1)))

            step_match = re.search(r"'step':\s*(\d+)", logs)
            if step_match:
                metrics.current_step = int(step_match.group(1))

            # Parse loss
            loss_match = re.search(r"'loss':\s*([\d.]+)", logs)
            if loss_match:
                metrics.loss = float(loss_match.group(1))

            # Parse learning rate
            lr_match = re.search(r"'learning_rate':\s*([\d.e-]+)", logs)
            if lr_match:
                metrics.learning_rate = float(lr_match.group(1))

            # Parse grad_norm
            grad_match = re.search(r"'grad_norm':\s*([\d.]+)", logs)
            if grad_match:
                metrics.grad_norm = float(grad_match.group(1))

            # Parse throughput
            throughput_match = re.search(r"([\d.]+)\s+samples?/s", logs, re.IGNORECASE)
            if throughput_match:
                metrics.samples_per_second = float(throughput_match.group(1))

            metrics.last_updated = datetime.now()

            return metrics if metrics.loss is not None else None

        except Exception:
            return None

    def _parse_evaluation_metrics(self, logs: str) -> Optional[EvaluationMetrics]:
        """
        Parse evaluation metrics from logs

        Args:
            logs: Log output from training job

        Returns:
            EvaluationMetrics object or None if not found
        """
        import re

        metrics = EvaluationMetrics()

        try:
            # Parse eval loss: 'eval_loss': 0.312
            eval_loss_match = re.search(r"'eval_loss':\s*([\d.]+)", logs)
            if eval_loss_match:
                metrics.eval_loss = float(eval_loss_match.group(1))

            # Parse eval accuracy
            eval_acc_match = re.search(r"'eval_accuracy':\s*([\d.]+)", logs)
            if eval_acc_match:
                metrics.eval_accuracy = float(eval_acc_match.group(1))

            # Parse perplexity
            perplexity_match = re.search(
                r"'eval_(?:runtime_)?perplexity':\s*([\d.]+)", logs
            )
            if perplexity_match:
                metrics.eval_perplexity = float(perplexity_match.group(1))

            # Parse eval samples
            eval_samples_match = re.search(r"'eval_samples':\s*(\d+)", logs)
            if eval_samples_match:
                metrics.eval_samples = int(eval_samples_match.group(1))

            metrics.last_evaluated = datetime.now()

            return metrics if metrics.eval_loss is not None else None

        except Exception:
            return None

    def _fetch_gpu_metrics(self, job_name: str) -> Optional[GPUMetrics]:
        """
        Fetch GPU utilization metrics from running cluster

        Args:
            job_name: Name of the job/cluster

        Returns:
            GPUMetrics object or None if unavailable
        """
        try:
            # Run nvidia-smi on the cluster
            cmd = [
                "sky",
                "exec",
                job_name,
                """nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw
                --format=csv,noheader,nounits""",
            ]
            result = self._run_sky_command(cmd, check=False)

            if result.returncode != 0:
                return None

            # Parse output: 95, 22345, 24576, 76, 285
            output = result.stdout.strip()
            if not output:
                return None

            parts = [p.strip() for p in output.split(",")]
            if len(parts) < 5:
                return None

            metrics = GPUMetrics(
                gpu_utilization=float(parts[0]),
                gpu_memory_used=float(parts[1]) / 1024,  # Convert MB to GB
                gpu_memory_total=float(parts[2]) / 1024,
                gpu_temperature=float(parts[3]),
                power_usage=float(parts[4]),
                last_updated=datetime.now(),
            )

            return metrics

        except Exception:
            return None

    def get_job_metrics(self, job_name: str) -> JobInfo:
        """
        Get job status with detailed metrics

        Args:
            job_name: Name of the job

        Returns:
            JobInfo with training, evaluation, and GPU metrics
        """
        # Get base job info
        job_info = self.get_job_status(job_name)

        # Only fetch metrics if job is running
        if job_info.status == JobStatus.RUNNING:
            try:
                # Get recent logs to parse metrics
                logs = self.get_job_logs(job_name, tail=500)

                # Parse training metrics
                job_info.training_metrics = self._parse_training_metrics(logs)

                # Parse evaluation metrics
                job_info.evaluation_metrics = self._parse_evaluation_metrics(logs)

                # Fetch GPU metrics
                job_info.gpu_metrics = self._fetch_gpu_metrics(job_name)

            except Exception:  # nosec B110
                pass  # Metrics are optional, continue without them

        return job_info
