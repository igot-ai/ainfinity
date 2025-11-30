#!/usr/bin/env python3
"""
Example client for AIFininity Training Service
Demonstrates how to use the API to manage training jobs
"""

import time
from typing import Any, Dict

import requests


class TrainingClient:
    """Client for interacting with AIFininity Training Service"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def launch_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Launch a new training job"""
        response = requests.post(f"{self.base_url}/api/v1/jobs", json=job_config, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status of a job"""
        response = requests.get(f"{self.base_url}/api/v1/jobs/{job_name}", timeout=10)
        response.raise_for_status()
        return response.json()

    def list_jobs(self) -> Dict[str, Any]:
        """List all jobs"""
        response = requests.get(f"{self.base_url}/api/v1/jobs", timeout=10)
        response.raise_for_status()
        return response.json()

    def stop_job(self, job_name: str) -> Dict[str, Any]:
        """Stop a running job"""
        response = requests.delete(f"{self.base_url}/api/v1/jobs/{job_name}", timeout=10)
        response.raise_for_status()
        return response.json()

    def get_logs(self, job_name: str, tail: int = 100) -> Dict[str, Any]:
        """Get job logs"""
        response = requests.get(f"{self.base_url}/api/v1/jobs/{job_name}/logs", params={"tail": tail}, timeout=10)
        response.raise_for_status()
        return response.json()

    def delete_job(self, job_name: str) -> Dict[str, Any]:
        """Delete job from database"""
        response = requests.delete(f"{self.base_url}/api/v1/jobs/{job_name}/delete", timeout=10)
        response.raise_for_status()
        return response.json()

    def wait_for_job(self, job_name: str, check_interval: int = 30, timeout: int = 3600) -> str:
        """
        Wait for a job to complete

        Args:
            job_name: Name of the job
            check_interval: Seconds between status checks
            timeout: Maximum seconds to wait

        Returns:
            Final job status
        """
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_name} did not complete within {timeout} seconds")

            status_response = self.get_job_status(job_name)
            status = status_response["job_info"]["status"]

            print(f"Job {job_name} status: {status}")

            if status in ["completed", "failed", "stopped"]:
                return status

            time.sleep(check_interval)


def main():
    """Example usage"""

    # Initialize client
    client = TrainingClient("http://localhost:8000")

    # Example 1: Launch a simple training job
    print("\n=== Example 1: Launch Training Job ===")
    job_config = {
        "job_name": "qwen3-orca-demo",
        "resources": {
            "infra": "vast",
            "accelerators": "RTX3090:1",
            "disk_size": 100,
            "image_id": "docker:nvidia/cuda:12.1.1-devel-ubuntu20.04",
        },
        "training": {
            "config_file": "finetuning",
            "dataset": "orca_chat",
            "model": "qwen3",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "learning_rate": 5e-05,
            "accelerate_config": "ds2_config.yaml",
        },
        "detach": True,
        "down": True,
    }

    try:
        result = client.launch_job(job_config)
        print(f"✓ Job launched: {result['message']}")
    except requests.exceptions.HTTPError as e:
        print(f"✗ Failed to launch job: {e}")
        return

    # Example 2: Check job status
    print("\n=== Example 2: Check Job Status ===")
    try:
        status = client.get_job_status("qwen3-orca-demo")
        print(f"Job status: {status['job_info']['status']}")
    except requests.exceptions.HTTPError as e:
        print(f"✗ Failed to get status: {e}")

    # Example 3: List all jobs
    print("\n=== Example 3: List All Jobs ===")
    try:
        jobs = client.list_jobs()
        print(f"Total jobs: {jobs['total']}")
        for job in jobs["jobs"]:
            print(f"  - {job['job_name']}: {job['status']}")
    except requests.exceptions.HTTPError as e:
        print(f"✗ Failed to list jobs: {e}")

    # Example 4: Get logs
    print("\n=== Example 4: Get Job Logs ===")
    try:
        logs = client.get_logs("qwen3-orca-demo", tail=20)
        print("Recent logs:")
        print(logs["logs"][:500] + "..." if len(logs["logs"]) > 500 else logs["logs"])
    except requests.exceptions.HTTPError as e:
        print(f"✗ Failed to get logs: {e}")

    # Example 5: Launch job with extra overrides
    print("\n=== Example 5: Launch with Custom Overrides ===")
    custom_job_config = {
        "job_name": "qwen3-custom-demo",
        "resources": {"infra": "vast", "accelerators": "A100:2", "disk_size": 200, "memory": "32+"},
        "training": {
            "config_file": "finetuning",
            "dataset": "orca_chat",
            "model": "qwen3",
            "output_dir": "./custom-checkpoints",
            "accelerate_config": "fsdp_config.yaml",
            "extra_args": {
                "training_arguments.gradient_accumulation_steps": "8",
                "training_arguments.logging_steps": "5",
            },
        },
        "detach": True,
        "down": False,  # Keep cluster running
    }

    try:
        result = client.launch_job(custom_job_config)
        print(f"✓ Custom job launched: {result['message']}")
    except requests.exceptions.HTTPError as e:
        print(f"✗ Failed to launch custom job: {e}")

    # Example 6: Wait for job completion (commented out to not block)
    # print("\n=== Example 6: Wait for Job Completion ===")
    # try:
    #     final_status = client.wait_for_job("qwen3-orca-demo", check_interval=30)
    #     print(f"Job completed with status: {final_status}")
    # except TimeoutError as e:
    #     print(f"✗ {e}")

    # Example 7: Stop a job
    print("\n=== Example 7: Stop Job ===")
    print("(Commented out - uncomment to actually stop the job)")
    # try:
    #     result = client.stop_job("qwen3-orca-demo")
    #     print(f"✓ Job stopped: {result['message']}")
    # except requests.exceptions.HTTPError as e:
    #     print(f"✗ Failed to stop job: {e}")


if __name__ == "__main__":
    main()
