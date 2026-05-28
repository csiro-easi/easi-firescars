# Engineering Plan: OLMo Earth Firescar Fine-tuning

## 1. Overview

This document translates the [Science Plan](SCIENCE_PLAN.md) into an implementation plan for the `easi-firescars` codebase. The system supports both single-GPU training (interactive development on a JupyterLab GPU server) and multi-GPU distributed training (production runs via Dask DDP or Argo Workflows).

## 2. Repository Structure

```
easi-firescars/
├── SCIENCE_PLAN.md
├── ENGINEERING_PLAN.md
├── configs/
│   ├── finetune.yaml            # Default training config (single-GPU)
│   └── finetune_ddp.yaml        # Multi-GPU DDP overrides
├── firescars/
│   ├── __init__.py
│   ├── train.py                 # Training entrypoint (single & multi-GPU)
│   ├── model.py                 # OLMo Earth encoder + UPerNet decoder
│   ├── dataset.py               # Zarr chip dataset + DALI pipeline
│   ├── loss.py                  # BCE + Dice combined loss
│   ├── checkpoint.py            # S3 checkpoint save/load
│   └── evaluate.py              # Metric computation and reporting
├── notebooks/
│   ├── data_prep.ipynb          # Distributed chip generation
│   ├── train_single_gpu.ipynb   # Interactive single-GPU training
│   ├── train_multi_gpu.ipynb    # Launch DDP via Dask
│   └── inference.ipynb          # Run predictions on new imagery
├── workflows/
│   ├── data_prep.yaml           # Argo: chip generation
│   ├── finetune.yaml            # Argo: single-GPU training
│   └── finetune_ddp.yaml       # Argo: multi-GPU DDP training
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_loss.py
└── pyproject.toml
```

## 3. Configuration System

A single YAML config drives both execution modes. The multi-GPU config inherits from the base and overrides distributed settings.

### 3.1 Base Config (`configs/finetune.yaml`)

```yaml
model:
  encoder: olmo_earth_pretrain
  decoder: upernet
  num_classes: 1
  input_bands: 6
  input_size: 224

data:
  chips_path: s3://easi-dc-data/staging/olmoearth_firescars_chips/
  metadata_path: s3://easi-dc-data/staging/olmoearth_firescars_chips/metadata.json
  batch_size: 16
  num_workers: 4
  use_dali: false

training:
  epochs: 30
  lr: 1.0e-4
  min_lr: 1.0e-6
  warmup_steps: 500
  weight_decay: 0.01
  encoder_lr_multiplier: 0.1
  early_stopping:
    patience: 5
    monitor: val_loss

loss:
  bce_weight: 0.5
  dice_weight: 0.5

augmentation:
  random_flip: true
  random_rotate90: true
  colour_jitter: true
  cutmix: true

checkpoint:
  path: s3://easi-dc-data/staging/olmoearth_firescars_checkpoints/
  save_every_epoch: true
  resume: true

distributed:
  enabled: false
  backend: nccl
  num_gpus: 1
```

### 3.2 Multi-GPU Override (`configs/finetune_ddp.yaml`)

```yaml
_base_: finetune.yaml

data:
  use_dali: true
  num_workers: 2  # DALI handles prefetch; fewer CPU workers needed

distributed:
  enabled: true
  num_gpus: 4

checkpoint:
  path: s3://easi-dc-data/staging/olmoearth_firescars_checkpoints/ddp/
```

## 4. Training Entrypoint

`firescars/train.py` is the single entrypoint for both modes. Execution mode is determined by `distributed.enabled` in the config.

### 4.1 Single-GPU Path

```
python -m firescars.train --config configs/finetune.yaml
```

- Runs directly on the current GPU (e.g., JupyterLab GPU server on `easi-user-gpu-node-pool`).
- Standard PyTorch DataLoader with optional DALI.
- Checkpoints to S3 via `firescars.checkpoint`.

### 4.2 Multi-GPU Path (Dask DDP)

Launched from a notebook or script that provisions Dask GPU workers:

```python
from easi_tools.dask_ddp.supervise import run_training_with_retries
from easi_tools.dask_ddp.dask_ddp import run

run_training_with_retries(
    run_fn=run,
    train_script="firescars/train.py",
    config="configs/finetune_ddp.yaml",
    max_retries=3,
    checkpoint_path="s3://easi-dc-data/staging/olmoearth_firescars_checkpoints/ddp/",
)
```

When `distributed.enabled=true`, `train.py`:
1. Reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` from environment (set by `easi_tools.dask_ddp`).
2. Initialises `torch.distributed` with NCCL backend.
3. Wraps model in `DistributedDataParallel`.
4. Uses `DistributedSampler` (or DALI sharding) to partition data across ranks.
5. Only rank 0 writes checkpoints and logs metrics.
6. Scales effective learning rate linearly: `lr = base_lr * world_size`.

### 4.3 Execution Flow

```
┌─────────────────────────────────────────────────────┐
│                  firescars/train.py                  │
├─────────────────────────────────────────────────────┤
│  1. Load config (YAML)                              │
│  2. If distributed.enabled:                         │
│       init_process_group(backend="nccl")            │
│       model = DDP(model, device_ids=[local_rank])   │
│       sampler = DistributedSampler(dataset)         │
│       lr *= world_size                              │
│     Else:                                           │
│       model = model.cuda()                          │
│  3. Build optimizer (AdamW, encoder LR multiplier)  │
│  4. Build scheduler (cosine annealing + warmup)     │
│  5. Resume from checkpoint if exists                │
│  6. Training loop:                                  │
│       for epoch in range(start_epoch, max_epochs):  │
│         train_one_epoch()                           │
│         val_metrics = validate()                    │
│         checkpoint (rank 0 only)                    │
│         early_stopping check                        │
│  7. Export model_best.pth to final S3 path          │
└─────────────────────────────────────────────────────┘
```

## 5. Module Design

### 5.1 `firescars/model.py`

- Loads OLMo Earth pretrained encoder weights.
- Attaches UPerNet decoder head (single output channel, sigmoid activation).
- Exposes `freeze_encoder()` / `unfreeze_encoder()` for ablation.
- Applies encoder LR multiplier via parameter groups.

### 5.2 `firescars/dataset.py`

Two data loading backends:

| Backend | When | Description |
|---------|------|-------------|
| PyTorch DataLoader | `use_dali: false` | Zarr random-access via fsspec. CPU augmentation with torchvision. |
| NVIDIA DALI | `use_dali: true` | GPU-decoded pipeline. Reads from Zarr/S3. Handles augmentation on GPU. |

Both backends support `DistributedSampler` / DALI shard_id for multi-GPU partitioning.

### 5.3 `firescars/loss.py`

```python
loss = bce_weight * F.binary_cross_entropy_with_logits(pred, target) \
     + dice_weight * dice_loss(pred, target)
```

### 5.4 `firescars/checkpoint.py`

- `save_checkpoint(state, path, is_best)` → writes `model_latest.pth`; copies to `model_best.pth` if `is_best`.
- `load_checkpoint(path)` → returns state dict, optimizer state, epoch, best metric.
- All I/O via boto3 to S3. Atomic writes using multipart upload.
- In DDP mode, only rank 0 saves; all ranks load on resume.

### 5.5 `firescars/evaluate.py`

- Computes IoU, F1, precision, recall on validation/test set.
- Supports per-bioregion breakdown using chip metadata.
- Outputs JSON metrics file alongside model checkpoint.

## 6. Infrastructure

### 6.1 Single-GPU (Interactive)

| Setting | Value |
|---------|-------|
| Server | JupyterLab GPU (hub.csiro.easi-eo.solutions) |
| Node pool | `easi-user-gpu-node-pool` (on-demand) |
| Instance | g4dn.4xlarge (1× T4, 16 GB VRAM, 16 vCPU, 64 GB RAM) |
| Use case | Development, debugging, hyperparameter search |

No special scheduling config needed — the JupyterLab spawner handles GPU node selection.

### 6.2 Multi-GPU (Dask DDP)

| Setting | Value |
|---------|-------|
| Orchestrator | Dask cluster with GPU workers |
| Node pool | `dask-gpu-worker-node-pool` (on-demand) |
| Workers | 1–4 (each with 1× T4) |
| Resources per worker | 7 CPU, 62 Gi memory, 1 GPU |
| Toleration | `nvidia.com/gpu Exists NoSchedule` |
| Node affinity | `eks.amazonaws.com/instance-gpu-manufacturer: nvidia` |
| Use case | Full training runs, production |

Dask cluster provisioned from notebook:

```python
from dask_kubernetes.operator import KubeCluster

cluster = KubeCluster(
    name="firescars-train",
    image="<easi-ml-image>:latest",
    n_workers=4,
    resources={
        "requests": {"cpu": "7", "memory": "62Gi", "nvidia.com/gpu": "1"},
        "limits": {"memory": "62Gi", "nvidia.com/gpu": "1"},
    },
    tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
    affinity={
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{
                    "matchExpressions": [{
                        "key": "eks.amazonaws.com/instance-gpu-manufacturer",
                        "operator": "In",
                        "values": ["nvidia"],
                    }]
                }]
            }
        }
    },
)
```

### 6.3 Multi-GPU (Argo Workflow)

For unattended production runs, the Argo workflow template launches the Dask cluster and training script:

```yaml
# workflows/finetune_ddp.yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: olmoearth-firescars-finetune-ddp
spec:
  entrypoint: ddp-train
  templates:
    - name: ddp-train
      container:
        image: <easi-ml-image>:latest
        command: [python, -m, firescars.train]
        args: ["--config", "configs/finetune_ddp.yaml"]
        resources:
          requests:
            cpu: "7"
            memory: "62Gi"
            nvidia.com/gpu: "1"
          limits:
            memory: "62Gi"
            nvidia.com/gpu: "1"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: eks.amazonaws.com/instance-gpu-manufacturer
                    operator: In
                    values:
                      - nvidia
```

Submit:
```bash
export ARGO_SERVER='argo.csiro.easi-eo.solutions:443'
export ARGO_SECURE=true
export ARGO_HTTP1=true
export ARGO_TOKEN=$(argo auth token)
argo submit -n $ARGO_NAMESPACE workflows/finetune_ddp.yaml
```

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single entrypoint for both modes | Reduces code duplication; config flag switches behaviour |
| Linear LR scaling for multi-GPU | Standard DDP practice to maintain effective batch statistics |
| DALI optional, not required | Single-GPU dev works without DALI; multi-GPU benefits from GPU-accelerated I/O |
| Rank 0 checkpointing only | Avoids write contention on S3; all ranks load same checkpoint on resume |
| Separate checkpoint paths per mode | Prevents single-GPU experiments from overwriting DDP production checkpoints |
| `easi_tools.dask_ddp` for orchestration | Provides fault tolerance (retry + resume) without custom infra code |
| Config inheritance (`_base_`) | DDP config only specifies deltas, reducing drift |

## 8. Fault Tolerance

| Failure Mode | Recovery |
|--------------|----------|
| Worker preemption (Dask DDP) | `run_training_with_retries` restarts from `model_latest.pth` |
| OOM on single GPU | Reduce `batch_size` in config; gradient accumulation fallback |
| S3 write failure | Retry with exponential backoff in `checkpoint.py` |
| NCCL timeout (multi-GPU) | `run_training_with_retries` catches and restarts full job |
| Stale checkpoint (incompatible model) | Version key in checkpoint dict; skip load on mismatch, warn user |

## 9. Testing Strategy

| Layer | Tool | What |
|-------|------|------|
| Unit | pytest | Model forward pass shapes, loss computation, checkpoint round-trip |
| Integration | pytest + moto | Dataset loading from mock S3, config parsing |
| Training smoke | Single epoch on 100 chips | Verifies full pipeline end-to-end (single-GPU) |
| DDP smoke | 2-worker Dask cluster, 1 epoch | Verifies gradient sync, distributed sampler, rank-0 checkpointing |

## 10. Implementation Phases

| Phase | Tasks | Depends On |
|-------|-------|------------|
| **P1: Core modules** | `model.py`, `loss.py`, `dataset.py` (PyTorch DataLoader), `checkpoint.py` | — |
| **P2: Single-GPU training** | `train.py` (single-GPU path), `configs/finetune.yaml`, unit tests | P1 |
| **P3: Multi-GPU training** | DDP logic in `train.py`, DALI pipeline in `dataset.py`, `configs/finetune_ddp.yaml` | P2 |
| **P4: Evaluation** | `evaluate.py`, per-bioregion metrics, comparison baselines | P2 |
| **P5: Workflows & notebooks** | Argo YAML, Jupyter notebooks for interactive and production use | P3, P4 |
| **P6: Documentation** | README update, inference notebook, evaluation report template | P5 |
