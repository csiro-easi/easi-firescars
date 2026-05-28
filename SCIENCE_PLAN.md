# Science Plan: Fine-tuning OLMo Earth on Australian Firescars

## 1. Objective

Fine-tune the OLMo Earth pretrained geospatial foundation model for binary burned-area segmentation using Australian firescar maps from the Digital Atlas of Australia. The goal is to produce a production-quality model that generalises across Australian biomes and fire regimes, deployable as an inference service on the EASI platform.

## 2. Scientific Rationale

### 2.1 Problem Statement

Accurate, timely mapping of burned areas is critical for fire management, carbon accounting, and ecosystem recovery monitoring across Australia. Existing firescar products rely on rule-based spectral indices (e.g., dNBR) that struggle with commission errors in spectrally similar land-cover transitions (e.g., harvest, drought stress). A learned segmentation model can capture spatial context and multi-band relationships that threshold approaches miss.

### 2.2 Why OLMo Earth

OLMo Earth is pretrained on large-scale, multi-sensor Earth observation data using self-supervised objectives. Its encoder has learned general-purpose spectral–spatial representations that transfer well to downstream remote sensing tasks. Fine-tuning on domain-specific firescar labels requires significantly less labelled data and compute than training from scratch, while achieving superior generalisation.

### 2.3 Study Area

Continental Australia — spanning tropical savannas (Northern Territory, Queensland), eucalypt forests (southeast), temperate grasslands, and semi-arid shrublands. This diversity tests model robustness across fire regimes with different burn severity, seasonality, and vegetation recovery rates.

## 3. Data

### 3.1 Source Labels

| Attribute | Value |
|-----------|-------|
| Product | Digital Atlas of Australia — firescar maps |
| S3 path | `s3://easi-dc-data/products-index/digital_atlas_firescars/` |
| Format | Cloud-Optimised GeoTIFF (COG) |
| Spatial coverage | Continental Australia |
| Temporal coverage | Multi-year historical record |
| Label schema | Binary mask — burned (1) / unburned (0) per pixel |
| Source imagery | Landsat 8/9, Sentinel-2 |

### 3.2 Input Imagery

Multispectral imagery co-registered with the firescar labels. Bands aligned to OLMo Earth pretrain channels:

| Band | Wavelength (µm) | Role |
|------|-----------------|------|
| Blue | 0.45–0.52 | Atmospheric scattering baseline |
| Green | 0.53–0.59 | Vegetation vigour |
| Red | 0.64–0.67 | Chlorophyll absorption |
| NIR | 0.85–0.88 | Canopy structure |
| SWIR1 | 1.57–1.65 | Moisture / burn signal |
| SWIR2 | 2.11–2.29 | Char / ash detection |

### 3.3 Chip Generation

Distributed data preparation using Dask workers (pattern from `easi_tools.dask_helpers`):

1. Query the Open Data Cube for paired imagery + firescar mask scenes.
2. Tile each scene into 224×224 pixel chips at native resolution.
3. Discard chips with >90% nodata or <1% burned pixels (class balance filter).
4. Normalise imagery to [0, 1] per-band using dataset-level statistics.
5. Write chips to Zarr/Icechunk store for efficient random-access training reads.
6. Apply stratified train/val/test split (70/15/15) by IBRA bioregion × fire season.

| Output | S3 Path |
|--------|---------|
| Training chips | `s3://easi-dc-data/staging/olmoearth_firescars_chips/` |
| Dataset metadata | `s3://easi-dc-data/staging/olmoearth_firescars_chips/metadata.json` |

## 4. Model Architecture & Training

### 4.1 Model Configuration

| Parameter | Value |
|-----------|-------|
| Base model | OLMo Earth pretrain (encoder) |
| Decoder | UPerNet segmentation head |
| Task | Binary semantic segmentation |
| Input size | 224 × 224 × 6 bands |
| Output | Single-channel probability map |

### 4.2 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Loss | 0.5 × BCE + 0.5 × Dice |
| Optimizer | AdamW (weight decay 0.01) |
| Learning rate | 1e-4, cosine annealing to 1e-6 |
| Warmup | 500 steps (linear) |
| Batch size | 16 per GPU |
| Epochs | 30 (early stopping, patience=5, monitor=val_loss) |
| Augmentation | Random flip, rotation (90°), colour jitter, cutmix |
| Encoder LR multiplier | 0.1 (slower fine-tuning of pretrained weights) |

### 4.3 Distributed Training on EASI

Training uses PyTorch DDP dispatched across Dask GPU workers via `easi_tools.dask_ddp`:

```python
from easi_tools.dask_ddp.supervise import run_training_with_retries
from easi_tools.dask_ddp.dask_ddp import run

run_training_with_retries(
    run_fn=run,
    train_script="firescars/train.py",
    config="configs/finetune.yaml",
    max_retries=3,
    checkpoint_path="s3://easi-dc-data/staging/olmoearth_firescars_checkpoints/",
)
```

Key patterns:
- **Fault tolerance**: Automatic retry with checkpoint resume (`model_latest.pth`).
- **Data loading**: NVIDIA DALI for GPU-accelerated decoding and augmentation.
- **Checkpointing**: `model_latest.pth` (every epoch) and `model_best.pth` (best val loss) saved to S3.

## 5. EASI Platform Execution

### 5.1 Compute Resources

| Resource | Data Prep | Training |
|----------|-----------|----------|
| Node pool | `dask-worker-node-pool` (spot) | `dask-gpu-worker-node-pool` (on-demand) |
| Instance | General purpose | GPU (NVIDIA T4, 16 GB VRAM) |
| GPUs | 0 | 1–4 (DDP across workers) |
| CPU | 4 cores/worker | 7 cores/worker |
| Memory | 32 Gi/worker | 62 Gi/worker |

### 5.2 Scheduling Configuration

```yaml
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

### 5.3 Argo Workflow

Production training runs are orchestrated via Argo Workflows for reproducibility:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: olmoearth-firescars-finetune
spec:
  entrypoint: train
  templates:
    - name: train
      container:
        image: <easi-ml-image>:latest
        command: [python, -m, firescars.train]
        args: ["--config", "configs/finetune.yaml"]
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

Submit via:
```bash
export ARGO_SERVER='argo.csiro.easi-eo.solutions:443'
export ARGO_SECURE=true
export ARGO_HTTP1=true
export ARGO_TOKEN=$(argo auth token)
argo submit -n $ARGO_NAMESPACE workflows/finetune.yaml
```

### 5.4 Data Access

| Purpose | Method | Path |
|---------|--------|------|
| Training chips (read) | S3 via Zarr/fsspec | `s3://easi-dc-data/staging/olmoearth_firescars_chips/` |
| Checkpoints (read/write) | boto3 | `s3://easi-dc-data/staging/olmoearth_firescars_checkpoints/` |
| Final model (write) | boto3 | `s3://easi-dc-data/products-index/olmoearth_firescars_model/` |

## 6. Evaluation

### 6.1 Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| IoU (burned class) | ≥ 0.75 | Primary metric |
| F1-score | ≥ 0.80 | Harmonic mean of precision/recall |
| Precision | ≥ 0.80 | Low commission error |
| Recall | ≥ 0.75 | Acceptable omission for small scars |

### 6.2 Evaluation Protocol

1. Evaluate on held-out test set (15% of chips) spanning all bioregions and fire seasons.
2. Report per-bioregion breakdown to identify regional weaknesses.
3. Compare against baseline: OLMo Earth encoder with frozen weights + linear head (no fine-tuning).
4. Compare against dNBR thresholding on the same test chips.
5. Qualitative assessment: visual inspection of predictions on 50 randomly sampled scenes covering edge cases (partial cloud, mixed severity, regrowth).

### 6.3 Failure Criteria

If IoU < 0.70 after full training:
- Investigate class imbalance (increase burned-chip sampling weight).
- Experiment with focal loss or boundary-aware loss.
- Consider unfreezing more encoder layers.

## 7. Outputs

| Deliverable | Location |
|-------------|----------|
| Fine-tuned model weights | `s3://easi-dc-data/products-index/olmoearth_firescars_model/` |
| Training config & code | This repository (`easi-firescars`) |
| Evaluation report | `docs/evaluation_report.md` |
| Inference notebook | `notebooks/inference.ipynb` |
| Argo Workflow definitions | `workflows/` |

## 8. Timeline

| Phase | Duration | Key Activities |
|-------|----------|----------------|
| Data preparation | 1 week | ODC queries, chip generation, Zarr write, QA |
| Training & tuning | 2 weeks | DDP training, hyperparameter sweeps, ablations |
| Evaluation & reporting | 1 week | Test-set metrics, per-bioregion analysis, documentation |
| **Total** | **4 weeks** | |

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Label noise in Digital Atlas product | Degraded precision | Visual QA on random sample; exclude low-confidence scenes |
| Class imbalance (burned << unburned) | Poor recall | Oversampling burned chips; Dice loss component |
| Spot instance preemption during data prep | Job failure | Dask worker retry; idempotent Zarr writes |
| GPU node unavailability | Training delay | On-demand node pool; checkpoint resume |
| Domain shift across sensors (Landsat vs S2) | Reduced generalisation | Include both sensors in training; band harmonisation |
