# Engineering Plan: Burn Scar Fine-Tuning with granite-geospatial-uki

## Workflow Overview

```mermaid
flowchart TD
    A[Download ImpactMesh-Fire subset] --> B[Extract & verify data]
    B --> C[Identify Australian test events]
    C --> D[Train: frozen backbone]
    D --> E[Train: full fine-tuning]
    E --> F[Evaluate on test set + AU subset]
    F --> G[Visual inspection of Black Summer predictions]
```

## 1. Data Access

### 1.1 ImpactMesh-Fire dataset

| Property | Value |
|----------|-------|
| Source | HuggingFace `ibm-esa-geospatial/ImpactMesh-Fire` |
| Access method | `huggingface_hub` CLI with `--include` filter |
| Modalities downloaded | S2L2A, S1RTC, MASK (skip DEM) |
| Splits | train, val, test + split/*.txt |
| Estimated download | ~30 GB |
| Local path | `data/ImpactMesh-Fire/` |
| Sample format | GeoTIFF or Zarr per sample (TBD — to be confirmed on first extraction) |

**Download command:**
```bash
source ~/venvs/burn-scar/bin/activate
hf download ibm-esa-geospatial/ImpactMesh-Fire \
  --repo-type dataset \
  --include "train/*.tar" \
  --include "val/*.tar" \
  --include "test/*.tar" \
  --include "split/*" \
  --exclude "*DEM*" \
  --local-dir data/ImpactMesh-Fire
```

### 1.2 Foundation model weights

| Property | Value |
|----------|-------|
| Source | HuggingFace `ibm-granite/granite-geospatial-uki` |
| Checkpoint file | `granite_geospatial_uki.pt` |
| Access method | `huggingface_hub.hf_hub_download()` → load with `PrithviViT` from terratorch |
| Architecture | PrithviViT (ViT-Base MAE, 12 layers, 768 dim, 8 in_chans, 3 pretrain frames) |
| Registry note | **Not registered** in terratorch 1.2.7 BACKBONE_REGISTRY; must be loaded manually |
| Size | ~400 MB |
| Cached at | `~/.cache/huggingface/` |

### 1.3 Data format (confirmed)

Inspection of the extracted dataset confirmed:
- **S2L2A format:** Zarr zip archives, shape `(4, 12, 256, 256)` int16. 4 timestamps, 12 bands (B01–B12). Band indices used: [1, 2, 3, 8, 10, 11] → B02, B03, B04, B8A, B11, B12.
- **S1RTC format:** Zarr zip archives, shape `(4, 2, 256, 256)` float16. Values are already in dB scale (typical range: -30 to 0). Some tiles contain NaN (missing SAR coverage).
- **MASK format:** GeoTIFF, shape `(1, 256, 256)` int8, binary 0/1.
- **Timestamps:** pre-month(0), pre-event(1), event(2), post-event(3). We use index 2 (event).
- **Spatial dimensions:** 256×256 at 10 m resolution (the model's positional embedding interpolation handles this natively despite img_size=224 in config).
- **Data completeness:** Not all split entries have files on disk (partial download). The dataset loader filters to samples with all three modalities present.

## 2. Processing Strategy

### 2.1 Environment setup

The EASI environment uses venvs linked to the base image packages. The `burn-scar` venv should already exist at `~/venvs/burn-scar/`. Dependencies are installed within this venv:

```bash
source ~/venvs/burn-scar/bin/activate
pip install --upgrade pip
pip install terratorch impactmesh huggingface_hub rasterio tensorboard
```

No Dask cluster is needed — this is a single-GPU PyTorch training workflow. The DataLoader handles data parallelism via `num_workers`.

### 2.2 Parallelisation unit

**Not applicable for training.** This is a standard single-GPU deep learning workflow:
- Data loading: PyTorch DataLoader with 4 worker processes
- Compute: Single GPU forward/backward pass
- No Dask, no DDP, no FSDP

### 2.3 Data pipeline

```mermaid
flowchart LR
    A[Zarr.zip / GeoTIFF on disk] --> B[PyTorch Dataset.__getitem__]
    B --> C[S2: int16 /10000, clip 0-1]
    B --> D[S1: float16 dB, clip -35 to 10, scale 0-1, NaN→0]
    C --> E[Concatenate 8 bands]
    D --> E
    E --> F[DataLoader batch=16, persistent_workers=True]
    F --> G[GPU]
```

**Normalisation details:**
- **S2 (bands 0–5):** `clip(arr.float32 / 10000, 0, 1)`. Source dtype: int16.
- **S1 (bands 6–7):** Data is already dB (float16). Auto-detect: if `nanmedian > 0 and < 1` → linear power, apply `10*log10`; otherwise treat as dB. Clip to [-35, 10], scale to [0, 1] via `(arr + 35) / 45`. Replace NaN with 0.0.
- **Data availability:** `_filter_available()` at Dataset init checks S2L2A, S1RTC, and MASK files exist for each sample.

**Per-sample memory:** 8 bands × 256 × 256 × 4 bytes (float32) = ~2.1 MB
**Per-batch memory:** 16 × 2.1 MB = ~33 MB (negligible)

### 2.4 Model architecture

The granite-geospatial-uki model uses the `PrithviViT` architecture from terratorch. It is **not registered** in the terratorch 1.2.7 backbone registry, so it is loaded directly from the HuggingFace checkpoint.

```
PrithviViT backbone (loaded from ibm-granite/granite-geospatial-uki .pt checkpoint)
    Config: img_size=224, num_frames=3, patch_size=[1,16,16], in_chans=8,
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4
    Checkpoint keys: encoder only (decoder/mask_token keys filtered out)
    Forward: input (B, 8, 1, 256, 256), mask_ratio=0.0
    Output: (B, 1+256, 768) — cls token + 16×16 spatial tokens
    pos_embed interpolation handles num_frames=1 at inference despite num_frames=3 at pretrain

Reshape: drop CLS token → (B, 768, 16, 16) spatial features

Decoder:
    Conv2d(768→256) + BN + ReLU + Upsample ×2   → (B, 256, 32, 32)
    Conv2d(256→128) + BN + ReLU + Upsample ×2   → (B, 128, 64, 64)
    Conv2d(128→64)  + BN + ReLU + Upsample ×4   → (B, 64, 256, 256)
    Conv2d(64→2, 1×1)                            → (B, 2, 256, 256)

Bilinear interpolate to input resolution if needed
```

**Total params:** ~88.8M. **Trainable (Phase 1):** ~2.1M (decoder only).

### 2.5 Training configuration

| Parameter | Phase 1 (frozen) | Phase 2 (unfrozen) |
|-----------|-------------------|---------------------|
| Epochs | 10 | 5 |
| Batch size | 16 | 8 |
| LR (decoder) | 1e-3 | 1e-4 |
| LR (backbone) | — | 1e-5 |
| Scheduler | CosineAnnealing | CosineAnnealing |
| Loss | Dice + BCE (50/50) | Dice + BCE (50/50) |
| Precision | fp16 (torch.amp) | fp16 (torch.amp) |
| Grad clip | max_norm=1.0 | max_norm=1.0 |
| Weight decay | 1e-4 | 1e-4 |

### 2.6 Evaluation

Metrics computed on the full test set and separately on samples matching `EMSR408*` (Australian Black Summer):
- IoU (burn scar class)
- Loss (Dice + BCE)

## 3. Resource Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | 1× (24+ GB VRAM recommended, e.g. A10, V100, A100) |
| GPU memory peak | ~12 GB (Phase 1), ~18 GB (Phase 2 with full model gradients) |
| System RAM | 32 GB |
| Disk | ~35 GB (30 GB dataset + 5 GB checkpoints/logs) |
| Training time | ~2-3 hrs Phase 1, ~2-3 hrs Phase 2 |
| Dask cluster | Not required |

## 4. Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|-----------|--------|
| No Australian events in dataset | Low | Cannot evaluate AU performance | Fall back to global metrics | ✅ Resolved — EMSR408 confirmed |
| S1 normalisation mismatch | Medium | Poor SAR feature learning | Inspect first sample; compare value ranges | ✅ Resolved — data is float16 dB, no log needed |
| S1 NaN values (missing SAR coverage) | Confirmed | NaN loss → training failure | Replace NaN with 0.0 after normalisation | ✅ Resolved |
| Incomplete data download | Confirmed | FileNotFoundError during training | `_filter_available()` skips missing samples | ✅ Resolved |
| granite_geospatial_uki not in terratorch registry | Confirmed | Model instantiation fails | Load PrithviViT directly + HF checkpoint | ✅ Resolved |
| OOM on Phase 2 | Medium | Training crashes | Reduce batch to 4; enable gradient checkpointing | Open |
| Band ordering wrong | Medium | Garbage predictions | Validated against granite-uki model card band order | ✅ Resolved |

## 5. Milestones (maps to science plan)

### M1: Dataset downloaded and verified
**Technical:** All 9 tar files extracted. Split .txt files present. Sample count ~22K across splits.
**Validation:** Load one random sample, display S2 RGB + S1 VV + mask side-by-side.

### M2: Australian events identified in test set
**Technical:** Grep split files for `EMSR408` (or other AU EMSR codes). Count samples.
**Validation:** ≥50 samples with AU-related EMSR codes in test split.

### M3: Training converges (Phase 1)
**Technical:** Loss decreasing, val IoU > 0.45 by epoch 10.
**Validation:** Plot loss and IoU curves.

### M4: Training converges (Phase 2)
**Technical:** Val IoU improves over Phase 1 best, no catastrophic forgetting.
**Validation:** Plot loss/IoU curves continuing from Phase 1.

### M5: Australian evaluation
**Technical:** IoU > 0.50 on AU test subset.
**Validation:** 4-panel visualisation (S2 RGB, S1 VV, ground truth, prediction) for 3-5 Black Summer events.

## 6. Code Structure

```
kiro-foundation-model/
├── README.md
├── science-plan.md
├── engineering-plan.md
├── workflow.ipynb              # Orchestration notebook (milestone-per-section)
├── modules/
│   ├── __init__.py
│   ├── data.py                # Dataset class, normalisation, split loading
│   ├── model.py               # GraniteUKIBurnScar model definition
│   ├── train.py               # Training loop, evaluation, checkpointing
│   └── visualisation.py       # Prediction overlays, training curves
└── tests/
    ├── conftest.py
    ├── test_data.py           # Normalisation correctness, band ordering
    ├── test_model.py          # Forward pass shape, backbone output handling
    └── test_train.py          # Loss computation, metric calculation
```

**Module responsibilities:**

| Module | Key Functions |
|--------|--------------|
| `modules/data.py` | `ImpactMeshFireDataset`, `normalize_s2()`, `normalize_s1()`, `get_split_samples()`, `identify_australian_events()` |
| `modules/model.py` | `GraniteUKIBurnScar` (nn.Module with frozen/unfrozen backbone + decoder) |
| `modules/train.py` | `train_epoch()`, `val_epoch()`, `compute_iou()`, `dice_bce_loss()` |
| `modules/visualisation.py` | `plot_training_curves()`, `plot_prediction_panels()`, `display_sample()` |

**Test strategy:**

| Test File | What | Data |
|-----------|------|------|
| `test_data.py` | S1/S2 normalisation ranges, band concatenation shape | Synthetic arrays |
| `test_model.py` | Forward pass produces (B, 2, 224, 224) from (B, 8, 224, 224) input | Random tensors |
| `test_train.py` | Dice+BCE loss on known inputs, IoU metric correctness | Synthetic logits/masks |

## 7. Notebook Outline

| Cell | Type | Milestone | Purpose |
|------|------|-----------|---------|
| 1 | Markdown | — | Science plan summary, milestone overview |
| 2 | Code | — | Configuration: paths, hyperparams, feature flags, verify environment |
| 3 | Markdown | — | Quick-start tip about MAX_TRAIN_SAMPLES |
| 4 | Markdown | M1 | Context: dataset verification |
| 5 | Code | M1/M2 | Collect Australian samples, create 70/15/15 splits |
| 6 | Code | M1 | Visual check: display random sample |
| 7 | Markdown | M2 | Context: Australian events in test set |
| 8 | Code | M2 | Report AU sample counts, validate sufficiency |
| 9 | Markdown | M3 | Context: Phase 1 training |
| 10 | Code | M3 | Run Phase 1 training loop (frozen backbone) |
| 11 | Code | M3 | Plot Phase 1 training curves |
| 12 | Markdown | M4 | Context: Phase 2 training |
| 13 | Code | M4 | Run Phase 2 training loop (full fine-tuning) |
| 14 | Code | M4 | Plot Phase 2 training curves |
| 15 | Markdown | M5 | Context: AU evaluation |
| 16 | Code | M5 | Evaluate on Australian test set |
| 17 | Code | M5 | 4-panel prediction visualisation for Black Summer |
| 18 | Markdown | — | Summary: metrics table, next steps |
