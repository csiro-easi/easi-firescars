# Science Plan: Australian Burned Area Mapping via Multi-Model Distillation

## 1. Objective

Produce a lightweight, high-accuracy burned-area segmentation model for continental Australia by:

1. Fine-tuning three geospatial foundation models independently on Australian firescar data.
2. Distilling their complementary knowledge into a single compact student model.
3. Deploying the distilled model as a production inference service on the EASI platform.

The three teacher models are:

| Model | Source | Params | Input Bands | Pre-training Domain |
|-------|--------|--------|-------------|---------------------|
| **OLMo Earth** | AI2 / CSIRO | ~300M | 6 optical (Landsat/S2) | Global multi-sensor EO |
| **Prithvi-EO-2.0-300M** | NASA / IBM | 300M | 6 optical (HLS) | Global HLS imagery |
| **Granite-geospatial-uki** | IBM | ~100M | 8 (6 optical + 2 SAR) | US HLS + UK/Ireland S1+S2 |

## 2. Scientific Rationale

### 2.1 Problem Statement

Accurate, timely mapping of burned areas is critical for fire management, carbon accounting, and ecosystem recovery monitoring across Australia. Existing firescar products rely on rule-based spectral indices (e.g., dNBR) that struggle with commission errors in spectrally similar land-cover transitions (harvest, drought stress). A learned segmentation model captures spatial context and multi-band relationships that threshold approaches miss.

### 2.2 Why Multiple Foundation Models

Each model brings different strengths:

- **OLMo Earth**: Broad spectral–spatial representations from diverse global sensors; strong generalisation to unseen biomes.
- **Prithvi-EO-2.0**: Proven benchmark performance on HLS Burn Scars (87.5% IoU); robust temporal representations from masked autoencoding on HLS time series.
- **Granite-geospatial-uki**: Unique SAR capability (Sentinel-1 VV/VH) enabling burn detection under cloud/smoke — critical during active Australian fire events.

### 2.3 Why Distillation

- **Complementary errors**: Models trained on different pre-training data make different mistakes. An ensemble or distilled model reduces systematic bias.
- **Operational efficiency**: A single compact student model (≤50M params) is cheaper to deploy than running three large models.
- **SAR–optical fusion**: The student learns to exploit optical-only inputs while inheriting SAR-informed decision boundaries from the Granite teacher.

### 2.4 Study Area

Continental Australia — spanning tropical savannas (Northern Territory, Queensland), eucalypt forests (southeast), temperate grasslands, and semi-arid shrublands. This diversity tests model robustness across fire regimes with different burn severity, seasonality, and vegetation recovery rates.

## 3. Data

### 3.1 Australian Training Data — Digital Atlas Firescars

| Attribute | Value |
|-----------|-------|
| Product | Digital Atlas of Australia — firescar maps |
| S3 path | `s3://easi-dc-data/products-index/digital_atlas_firescars/` |
| Format | Cloud-Optimised GeoTIFF (COG) |
| Spatial coverage | Continental Australia |
| Temporal coverage | Multi-year historical record |
| Label schema | Binary mask — burned (1) / unburned (0) |
| Source imagery | Landsat 8/9, Sentinel-2 |

### 3.2 Supplementary Data — ImpactMesh-Fire (Australian subset)

Used for the Granite model branch (SAR + optical):

| Attribute | Value |
|-----------|-------|
| Product | ImpactMesh-Fire (IBM/DLR/ESA) |
| Subset | EMSR408 — Black Summer 2019–2020, eastern NSW |
| Modalities | Sentinel-2 L2A, Sentinel-1 RTC, burn scar masks |
| Samples | ~2,877 (MGRS tile 56HKJ) |
| Resolution | 10 m |
| License | CC-BY 4.0 |

### 3.3 Supplementary Data — HLS Burn Scars

Used for Prithvi model benchmarking:

| Attribute | Value |
|-----------|-------|
| Product | HLS Burn Scars (Hugging Face) |
| Bands | 6 optical at 30 m |
| Benchmark IoU | 87.5% (published) |

### 3.4 Input Bands (Unified Optical)

All three models share a common 6-band optical core:

| Band | Wavelength (µm) | Role |
|------|-----------------|------|
| Blue | 0.45–0.52 | Atmospheric scattering baseline |
| Green | 0.53–0.59 | Vegetation vigour |
| Red | 0.64–0.67 | Chlorophyll absorption |
| NIR | 0.85–0.88 | Canopy structure |
| SWIR1 | 1.57–1.65 | Moisture / burn signal |
| SWIR2 | 2.11–2.29 | Char / ash detection |

Granite additionally uses VV and VH (C-band SAR).

### 3.5 Chip Generation

Distributed data preparation using Dask workers (`easi_tools.dask_helpers`):

1. Query the Open Data Cube for paired imagery + firescar mask scenes.
2. Tile into 224×224 pixel chips at native resolution.
3. Discard chips with >90% nodata or <1% burned pixels.
4. Normalise imagery to [0, 1] per-band using dataset-level statistics.
5. Write to Zarr/Icechunk store on S3.
6. Stratified train/val/test split (70/15/15) by IBRA bioregion × fire season.

| Output | S3 Path |
|--------|---------|
| Training chips | `s3://easi-dc-data/staging/firescars_chips/` |
| Dataset metadata | `s3://easi-dc-data/staging/firescars_chips/metadata.json` |

## 4. Phase 1: Independent Fine-Tuning

### 4.1 OLMo Earth (Branch: `feat/olmo-earth-train`)

| Parameter | Value |
|-----------|-------|
| Encoder | OLMo Earth pretrain |
| Decoder | UPerNet |
| Input | 224×224×6 bands |
| Loss | 0.5×BCE + 0.5×Dice |
| Optimizer | AdamW (lr=1e-4, cosine → 1e-6) |
| Encoder LR | ×0.1 |
| Epochs | 30 (early stop patience=5) |
| Batch size | 16/GPU |
| Data | Digital Atlas firescars (continental AU) |

### 4.2 Prithvi-EO-2.0 (Branch: `feat/prithvi-test`)

| Parameter | Value |
|-----------|-------|
| Encoder | Prithvi-EO-2.0-300M (ViT) |
| Decoder | UNet (channels: 512→256→128→64) |
| Input | 224×224×6 bands |
| Loss | Cross-entropy |
| Optimizer | AdamW (lr=1e-4, ReduceLROnPlateau) |
| Epochs | 5 quick-test / 50 full |
| Batch size | 8/GPU |
| Data | HLS Burn Scars + Digital Atlas firescars |
| Framework | TerraTorch |

### 4.3 Granite-geospatial-uki (Branch: `feat/granite-train`)

| Parameter | Value |
|-----------|-------|
| Encoder | Granite-geospatial-uki (PrithviViT, 12-layer) |
| Decoder | Conv upsampling (768→256→128→64→2) |
| Input | 224×224×8 bands (6 optical + VV, VH) |
| Loss | Dice + BCE |
| Training | 2-phase: frozen backbone → full fine-tune |
| Optimizer | AdamW (lr=1e-4) |
| Batch size | 8/GPU |
| Data | ImpactMesh-Fire Australian subset (EMSR408) |
| S1 norm | 10×log₁₀(σ₀), clip [-35,10], scale [0,1] |
| S2 norm | ÷10000, clip [0,1] |

## 5. Phase 2: Knowledge Distillation

### 5.1 Distillation Strategy

Train a compact student model using soft labels from the three fine-tuned teachers:

```
Student loss = α × Hard_loss(pred, ground_truth)
             + (1-α) × KD_loss(pred_soft, teacher_ensemble_soft)
```

Where:
- `teacher_ensemble_soft` = weighted average of teacher probability maps
- Temperature τ = 3.0 for soft targets
- α = 0.5 (equal weight hard/soft)

### 5.2 Teacher Weighting

Teachers are weighted by their validation IoU on the Australian test set:

```python
weights = softmax([iou_olmo, iou_prithvi, iou_granite] / temperature)
ensemble_prob = sum(w_i * teacher_i_prob for w_i in weights)
```

For samples without SAR data, the Granite teacher weight is redistributed to the other two.

### 5.3 Student Architecture

| Parameter | Value |
|-----------|-------|
| Encoder | EfficientNet-B3 (ImageNet pretrained) |
| Decoder | UNet (lightweight) |
| Input | 224×224×6 bands (optical only) |
| Parameters | ~12M (vs 300M per teacher) |
| Output | Single-channel probability map |

The student uses only optical bands for maximum operational flexibility (no SAR dependency at inference time).

### 5.4 Distillation Training

| Parameter | Value |
|-----------|-------|
| Loss | α×(BCE+Dice) + (1-α)×KL-div(soft) |
| α | 0.5 |
| Temperature | 3.0 |
| Optimizer | AdamW (lr=3e-4, cosine → 1e-6) |
| Epochs | 50 (early stop patience=10) |
| Batch size | 32/GPU |
| Augmentation | Random flip, rotation, colour jitter |

## 6. EASI Platform Execution

### 6.1 Compute Resources

| Phase | Node Pool | GPU | CPU | Memory |
|-------|-----------|-----|-----|--------|
| Data prep | `dask-worker-node-pool` (spot) | 0 | 4/worker | 32 Gi |
| Teacher training | `dask-gpu-worker-node-pool` (on-demand) | 1–4 T4 | 7/worker | 62 Gi |
| Distillation | `dask-gpu-worker-node-pool` (on-demand) | 1 T4 | 7/worker | 62 Gi |

### 6.2 Scheduling

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
              values: [nvidia]
```

### 6.3 Distributed Training

All teacher training uses DDP via `easi_tools.dask_ddp`:

```python
from easi_tools.dask_ddp.supervise import run_training_with_retries
from easi_tools.dask_ddp.dask_ddp import run

run_training_with_retries(
    run_fn=run,
    train_script="firescars/train.py",
    config="configs/finetune.yaml",
    max_retries=3,
    checkpoint_path="s3://easi-dc-data/staging/firescars_checkpoints/",
)
```

### 6.4 Argo Workflows

Production runs orchestrated via Argo for reproducibility:

```bash
export ARGO_SERVER='argo.csiro.easi-eo.solutions:443'
argo submit -n $ARGO_NAMESPACE workflows/train_teachers.yaml
argo submit -n $ARGO_NAMESPACE workflows/distill.yaml
```

### 6.5 Data Paths

| Purpose | S3 Path |
|---------|---------|
| Training chips | `s3://easi-dc-data/staging/firescars_chips/` |
| OLMo Earth checkpoints | `s3://easi-dc-data/staging/firescars_ckpt/olmo_earth/` |
| Prithvi checkpoints | `s3://easi-dc-data/staging/firescars_ckpt/prithvi/` |
| Granite checkpoints | `s3://easi-dc-data/staging/firescars_ckpt/granite/` |
| Distilled model | `s3://easi-dc-data/products-index/firescars_distilled_model/` |

## 7. Evaluation

### 7.1 Metrics

| Metric | Teacher Target | Student Target |
|--------|---------------|----------------|
| IoU (burned) | ≥ 0.75 | ≥ 0.72 |
| F1-score | ≥ 0.80 | ≥ 0.78 |
| Precision | ≥ 0.80 | ≥ 0.78 |
| Recall | ≥ 0.75 | ≥ 0.72 |

### 7.2 Evaluation Protocol

1. Evaluate all models on the same held-out Australian test set (15% of chips, stratified by bioregion).
2. Report per-bioregion breakdown for each teacher and the student.
3. Compare against baselines:
   - dNBR thresholding
   - Each teacher with frozen encoder + linear head (no fine-tuning)
   - Simple ensemble (average of teacher predictions)
4. Measure inference latency: student vs teachers vs ensemble.
5. Qualitative: visual inspection on 50 scenes covering edge cases (cloud, mixed severity, regrowth).

### 7.3 Ablation Studies

| Experiment | Purpose |
|------------|---------|
| Student without distillation (hard labels only) | Quantify distillation benefit |
| Single-teacher distillation (×3) | Identify most informative teacher |
| Varying α (0.3, 0.5, 0.7) | Optimal hard/soft balance |
| Varying temperature (1, 3, 5) | Soft target sharpness |
| Student with SAR input (8 bands) | Value of SAR at inference |

### 7.4 Failure Criteria

- If best teacher IoU < 0.60: investigate data pipeline, normalisation, label quality.
- If student IoU < teacher IoU − 0.10: increase student capacity or training duration.
- If student IoU < 0.65: consider using the best teacher directly instead of distilling.

## 8. Outputs

| Deliverable | Location |
|-------------|----------|
| Fine-tuned OLMo Earth weights | `s3://easi-dc-data/staging/firescars_ckpt/olmo_earth/model_best.pth` |
| Fine-tuned Prithvi weights | `s3://easi-dc-data/staging/firescars_ckpt/prithvi/model_best.pth` |
| Fine-tuned Granite weights | `s3://easi-dc-data/staging/firescars_ckpt/granite/model_best.pth` |
| Distilled student model | `s3://easi-dc-data/products-index/firescars_distilled_model/` |
| Training configs & code | This repository (`easi-firescars`) |
| Evaluation report | `docs/evaluation_report.md` |
| Inference notebook | `notebooks/inference.ipynb` |
| Argo Workflow definitions | `workflows/` |

## 9. Timeline

| Phase | Duration | Key Activities |
|-------|----------|----------------|
| Data preparation | 1 week | ODC queries, chip generation, Zarr write, QA |
| Teacher fine-tuning (×3 parallel) | 2 weeks | DDP training per model, hyperparameter sweeps |
| Distillation | 1 week | Generate soft labels, train student, ablations |
| Evaluation & reporting | 1 week | Test metrics, per-bioregion analysis, latency benchmarks |
| **Total** | **5 weeks** | |

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Label noise in Digital Atlas | Degraded precision | Visual QA; exclude low-confidence scenes |
| Class imbalance (burned << unburned) | Poor recall | Oversampling burned chips; Dice loss |
| Teacher disagreement on hard cases | Noisy soft labels | Confidence-weighted ensemble; discard low-agreement chips |
| Granite SAR unavailable for some scenes | Incomplete ensemble | Graceful fallback to 2-teacher average |
| Student capacity too low | IoU gap vs teachers | Scale to EfficientNet-B5 or use MobileViT |
| Spot preemption during data prep | Job failure | Dask retry; idempotent Zarr writes |
| GPU node unavailability | Training delay | On-demand pool; checkpoint resume |
| Domain shift (Landsat vs S2) | Reduced generalisation | Include both sensors; band harmonisation |
