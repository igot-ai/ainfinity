# SLM Fine-tuning Tracking & Monitoring Standard

This document outlines the standard practice for tracking and monitoring Small Language Model (SLM) fine-tuning jobs.

## 1. Tools
- **Orchestration**: SkyPilot (GCP target)
- **Fine-tuning**: Unsloth (optimized LoRA/QLoRA)
- **Tracking**: Weights & Biases (WandB) + Local Logging

## 2. Monitoring Standard (`monitoring.py`)
All training scripts must use the `ExperimentTracker` class from `monitoring.py`.

### Usage
```python
from monitoring import get_tracker

tracker = get_tracker(
    project_name="my-project",
    run_name="experiment-1",
    config={"param": "value"}
)

tracker.log_metrics({"loss": 0.5}, step=10)
tracker.finish()
```

### Metrics to Track
- **System Metrics**: GPU usage, Memory (handled automatically by WandB system monitoring).
- **Training Metrics**: Loss, Learning Rate, Epoch, Step time.
- **Artifacts**: Saved model checkpoints, dataset versions.

## 3. Deployment Standard (`sky.yaml`)
- Define resources explicitly (e.g., `accelerators: L4:1`).
- Use `setup` block for dependency installation.
- Pass secrets (WandB key) via environment variables.

## 4. Workflow
1. **Develop**: Test `train.py` locally or on a small instance.
2. **Launch**: `sky launch -c my-cluster sky.yaml --env WANDB_API_KEY=$WANDB_API_KEY`
3. **Monitor**: Check WandB dashboard and SkyPilot logs (`sky logs my-cluster`).
4. **Stop**: `sky down my-cluster`
