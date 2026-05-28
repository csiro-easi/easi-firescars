# EASI Fire Scars — Multi-Model Distillation

Australian burned area segmentation via multi-model knowledge distillation on the EASI platform.

## Overview

This project fine-tunes three geospatial foundation models on Australian firescar data, then distills their complementary knowledge into a single lightweight model for operational deployment.

### Teacher Models

| Model | Branch | Input | Key Strength |
|-------|--------|-------|--------------|
| OLMo Earth (~300M) | `feat/olmo-earth-train` | 6 optical bands | Broad multi-sensor generalisation |
| Prithvi-EO-2.0 (300M) | `feat/prithvi-test` | 6 optical bands | Proven HLS burn scar benchmark (87.5% IoU) |
| Granite-geospatial-uki (~100M) | `feat/granite-train` | 6 optical + 2 SAR | Cloud/smoke-penetrating SAR capability |

### Student Model

- **Architecture**: EfficientNet-B3 + UNet decoder (~12M params)
- **Input**: 6 optical bands (no SAR dependency at inference)
- **Training**: Knowledge distillation from teacher ensemble soft labels

## Repository Structure

```
easi-firescars/
├── SCIENCE_PLAN.md              # Unified science plan (this document)
├── README.md                    # This file
├── feat/olmo-earth-train        # OLMo Earth fine-tuning branch
├── feat/prithvi-test            # Prithvi-EO-2.0 fine-tuning branch
└── feat/granite-train           # Granite-geospatial-uki fine-tuning branch
```

Each feature branch contains its own training code, notebooks, and tests.

## Data

| Dataset | Purpose | Location |
|---------|---------|----------|
| Digital Atlas Firescars | Primary labels (continental AU) | `s3://easi-dc-data/products-index/digital_atlas_firescars/` |
| ImpactMesh-Fire (EMSR408) | SAR+optical for Granite | HuggingFace `ibm-esa-geospatial/ImpactMesh-Fire` |
| HLS Burn Scars | Prithvi benchmarking | HuggingFace |

## Platform Requirements

- **GPU**: NVIDIA T4 (16 GB VRAM) on `dask-gpu-worker-node-pool`
- **Training**: PyTorch DDP via `easi_tools.dask_ddp`
- **Orchestration**: Argo Workflows (`argo.csiro.easi-eo.solutions:443`)
- **Storage**: S3 (`s3://easi-dc-data/staging/firescars_*`)

## Quick Start

```bash
# Spawn GPU notebook server on EASI (hub.csiro.easi-eo.solutions)
# Select CUDA image, g4dn instance type, 1+ GPUs

# Clone and checkout a teacher branch
git clone git@github.com:csiro-easi/easi-firescars.git
cd easi-firescars
git checkout feat/olmo-earth-train  # or feat/prithvi-test, feat/granite-train
```

## Workflow

1. **Data Preparation** — Distributed chip generation from ODC (see `notebooks/data_prep.ipynb` on each branch)
2. **Teacher Fine-Tuning** — Train each model independently on its branch
3. **Knowledge Distillation** — Combine teacher soft labels into student training
4. **Evaluation** — Per-bioregion metrics, ablation studies, latency benchmarks

See [SCIENCE_PLAN.md](SCIENCE_PLAN.md) for full methodology.

## Target Metrics

| Metric | Teacher Target | Student Target |
|--------|---------------|----------------|
| IoU (burned) | ≥ 0.75 | ≥ 0.72 |
| F1-score | ≥ 0.80 | ≥ 0.78 |

## License

Internal CSIRO / EASI project.
