# Science Plan: Burn Scar Mapping with granite-geospatial-uki

## Research Objectives

Evaluate whether the granite-geospatial-uki foundation model — pre-trained on both optical (Sentinel-2) and SAR (Sentinel-1) imagery — can be fine-tuned to accurately map fire burn scars on the east coast of Australia.

The geographic focus is eastern Australia (New South Wales, Victoria, and Queensland), where eucalypt forests and coastal vegetation experience intense bushfire events. The temporal scope covers fire events in the ImpactMesh-Fire dataset (primarily 2019–2024), with particular attention to the 2019–2020 Black Summer fires for validation.

This is an exploratory evaluation of the model's transferability to Australian fire conditions. The model is trained using all 8 input bands (6 optical + 2 SAR) to leverage the cloud-penetrating capability of radar for burn scar detection under smoke and persistent cloud cover — conditions common during major Australian fire events.

## Input Data Sources

### Foundation model: granite-geospatial-uki

A Vision Transformer pre-trained using masked autoencoding on Harmonized Landsat-Sentinel 2 (HLS) imagery from the continental USA, followed by continual pre-training with HLS and Sentinel-1 imagery over the United Kingdom and Ireland. The model accepts 8 input bands:

- Blue, Green, Red, Narrow NIR, SWIR1, SWIR2 (optical, from HLS/Sentinel-2)
- VV, VH (C-band SAR backscatter, from Sentinel-1)

The model has approximately 100 million parameters and uses the same architecture as Prithvi-EO. Pre-trained weights are publicly available on HuggingFace (`ibm-granite/granite-geospatial-uki`).

### Training dataset: ImpactMesh-Fire

A large-scale multimodal, multitemporal dataset for wildfire mapping released by IBM, DLR, and the ESA Φ-lab. Key characteristics:

- **Modalities**: Sentinel-1 RTC (radiometrically terrain-corrected, VV and VH polarisations), Sentinel-2 Level-2A (surface reflectance), Copernicus DEM, and pixel-level burn scar masks
- **Annotations**: Pixel-level burn scar masks derived from Copernicus Emergency Management Service (EMS) activations — high-quality manual delineations by trained analysts
- **Scale**: Approximately 22,000 samples covering 200+ fire events globally
- **Temporal structure**: Four timestamps per event (one month before, immediately before, during/after event, one month after event)
- **Spatial coverage**: Global, including events where Copernicus EMS was activated. Australian fire events are present — EMSR408 (Black Summer 2019–2020, MGRS tile 56HKJ in SE Australia) appears across all splits
- **Resolution**: Sentinel-2 at 10 m, Sentinel-1 resampled to the same 10 m grid
- **License**: CC-BY 4.0
- **Access**: HuggingFace (`ibm-esa-geospatial/ImpactMesh-Fire`)

**Data subset for this project:** Download only the three modalities required for the 8-band model — Sentinel-1 RTC, Sentinel-2 L2A, and burn scar masks. Skip the Copernicus DEM (not needed for granite-geospatial-uki which was not pre-trained with elevation data). This reduces the download from ~58 GB to ~30 GB. The dataset is organised as one tar archive per modality per split (train/val/test), so selective download is straightforward:

```
hf download ibm-esa-geospatial/ImpactMesh-Fire --include "train/S2L2A.tar" "train/S1RTC.tar" "train/MASK.tar"
                                                         "val/S2L2A.tar" "val/S1RTC.tar" "val/MASK.tar"
                                                         "test/S2L2A.tar" "test/S1RTC.tar" "test/MASK.tar"
                                                         "split/*"
```

### Sentinel-1 normalisation convention

The granite-geospatial-uki model was pre-trained with Sentinel-1 backscatter normalised as 10×log₁₀(σ₀), clipped to the range [-35, 10] dB. The ImpactMesh-Fire dataset provides Sentinel-1 RTC data as **float16 values already in dB scale** (typical range: -30 to 0 dB). No log conversion is needed — only clipping to [-35, 10] and linear rescaling to [0, 1].

**NaN handling:** Some S1 tiles contain NaN values (missing SAR coverage, e.g. where orbit geometry provides no data). These are replaced with 0.0 after normalisation (equivalent to -35 dB, i.e. minimal backscatter). Approximately 15% of samples have partial NaN coverage in the S1 bands.

### Sentinel-2 normalisation convention

The model was pre-trained on HLS surface reflectance values. Sentinel-2 Level-2A reflectance values (typically stored as integers scaled by 10,000) should be divided by 10,000 to produce reflectance in the range [0, 1].

## Method / Processing Steps

### 1. Download and prepare the ImpactMesh-Fire dataset

Obtain the dataset from HuggingFace and extract all modalities (Sentinel-1 RTC, Sentinel-2 L2A, and burn scar masks). Use the "event" timestamp — the image acquired during or immediately after the fire — as the primary input for single-frame segmentation.

### 2. Prepare 8-band input chips

For each sample, construct an 8-band image chip by:
- Taking the 6 relevant Sentinel-2 bands (Blue, Green, Red, Narrow NIR, SWIR1, SWIR2) from int16 surface reflectance and normalising to [0, 1] by dividing by 10,000 and clipping
- Taking the 2 Sentinel-1 bands (VV, VH) — already stored in dB (float16) — clipping to [-35, 10] and rescaling to [0, 1]
- Replacing any NaN values in S1 bands with 0.0 (represents -35 dB / no signal)
- Concatenating into a single 8-band chip at 256×256 pixels (original tile size; centre-cropped or resized to 224×224 for the model)

**Data availability filtering:** Not all samples in the split files have complete data on disk (partial downloads or missing modalities). The dataset loader checks for existence of all three files (S2L2A, S1RTC, MASK) at initialisation and silently excludes incomplete samples.

### 3. Split data into training, validation, and test sets

Use the split files provided with ImpactMesh-Fire (train/val/test). Confirm that any Australian east-coast events (particularly Black Summer fires in NSW, Victoria, and Queensland) are placed in the test set so they can serve as an independent validation of the model's performance on Australian landscapes.

### 4. Fine-tune the model with the backbone initially frozen

Attach a convolutional decoder (upsampling network) to the frozen foundation model backbone. Train only the decoder to learn to map the model's pre-trained representations to burn scar masks. This phase establishes whether the existing features are sufficient for the task without risk of catastrophic forgetting.

### 5. Fine-tune with the backbone unfrozen

Unfreeze the backbone and continue training the entire model at a lower learning rate. This allows the model to adapt its internal representations to fire-specific spectral and SAR signatures in vegetation types not seen during pre-training (Australian eucalypt forests, tropical savanna).

### 6. Evaluate on the test set, with focus on Australian events

Compute segmentation accuracy metrics on the full test set and separately on the Australian east-coast subset. Compare performance between the two subsets to assess transferability.

## Expected Outputs

### Trained model weights

A fine-tuned version of granite-geospatial-uki with an attached segmentation decoder, capable of producing binary burn scar masks from 8-band (optical + SAR) input imagery at 10 m resolution.

### Evaluation metrics

For the full test set and the Australian east-coast subset separately:

- **Intersection over Union (IoU)** for the burn scar class — the primary metric. Expected range: 0.55–0.75 based on published results for similar tasks.
- **F1-score** (Dice coefficient) — expected range: 0.70–0.85
- **Precision and recall** — to understand whether errors are dominated by false positives (over-mapping) or false negatives (under-mapping)

### Prediction maps

For selected Black Summer fire events in eastern Australia, produce predicted burn scar masks overlaid on the input imagery. These should be visually inspectable to assess whether the model captures the spatial extent and boundaries of known burn scars.

### Training history

Loss and IoU curves across training epochs for both phases (frozen and unfrozen backbone), to confirm the model converged and did not overfit.

## Milestones and Validation Criteria

### Milestone 1: Dataset downloaded and verified

**What you should see:** The ImpactMesh-Fire dataset is extracted with three modalities present (Sentinel-1 RTC, Sentinel-2 L2A, and burn scar masks — DEM intentionally excluded). The total sample count is approximately 22,000 across train, validation, and test splits. Each sample has matching files across all three modalities. The total download is approximately 30 GB.

**Visual check:** Display a random sample showing the Sentinel-2 RGB composite, the Sentinel-1 VV backscatter, and the burn scar mask side by side. The burn scar region in the mask should correspond to visually darkened or changed areas in the imagery.

**Stop if:** Fewer than 10,000 samples extracted (incomplete download). Any modality is entirely missing (extraction error). Mask files are all zeros (wrong mask layer) or all ones (inverted mask).

### Milestone 2: Australian east-coast events identified in test set

**What you should see:** At least 2–5 fire events from eastern Australia are present in the dataset and allocated to the test split. These should include events from the 2019–2020 fire season if Copernicus EMS activations for Black Summer are in the dataset.

**Visual check:** Show the geographic locations of all fire events in the dataset on a map, with Australian east-coast events highlighted. Confirm they fall within NSW, Victoria, or Queensland.

**Stop if:** No Australian events are present in the dataset (the model can still be trained on global data, but the east-coast evaluation objective cannot be met — consider sourcing additional Australian reference data from Digital Earth Australia burnt area products). Fewer than 50 Australian test samples (insufficient for meaningful evaluation).

### Milestone 3: Model training converges (frozen backbone phase)

**What you should see:** Training loss decreases steadily over 15–20 epochs. Validation IoU improves from near-zero to above 0.45 within the first 10 epochs, indicating the decoder is successfully learning to interpret the backbone's features for burn scar mapping.

**Visual check:** Plot training loss and validation IoU curves across epochs. The curves should show clear improvement without erratic oscillation.

**Stop if:** Validation IoU remains below 0.30 after 15 epochs (the backbone features may not be suitable for this task, or the data pipeline has an error — check that bands are in the correct order and normalisation matches the model's pre-training). Loss does not decrease (learning rate too low, or gradient flow is blocked).

### Milestone 4: Model training converges (full fine-tuning phase)

**What you should see:** Starting from the best frozen-backbone checkpoint, validation IoU improves further — ideally from ~0.50 to above 0.60. Training is stable without sudden loss spikes that would indicate catastrophic forgetting.

**Visual check:** Same loss/IoU curves, continuing from where Phase 1 ended. Also show a few sample predictions from the validation set: the predicted mask should roughly align with the ground truth mask, capturing the main burn scar extent even if boundaries are imperfect.

**Stop if:** Validation IoU decreases compared to the frozen-backbone result (learning rate too high, causing forgetting — reduce it). Training loss explodes (gradient instability — reduce learning rate or add gradient clipping).

### Milestone 5: Evaluation on Australian east-coast fires

**What you should see:** IoU on the Australian east-coast test subset is above 0.50. This would indicate the model generalises to Australian landscapes despite being pre-trained on UK/Ireland data and trained primarily on global (non-Australian) fire events.

**Visual check:** For 3–5 Black Summer fire events, show the Sentinel-2 RGB, Sentinel-1 VV, ground truth mask, and predicted mask in a 4-panel figure. Assess visually:
- Does the prediction capture the overall fire extent?
- Are the boundaries reasonable (following vegetation boundaries, ridge lines)?
- Are there obvious false positives (non-burned areas mapped as burn)?

**Deep-dive — Black Summer, South Coast NSW:** If events from the Shoalhaven or Bega Valley region are in the test set, examine these in detail. These areas experienced near-complete canopy loss in eucalypt forests — the burn signal should be strong in both optical (low NIR, high SWIR) and SAR (reduced VH backscatter from canopy loss). Show the model's prediction alongside the Copernicus EMS reference mask for this area.

**Stop if:** IoU on Australian events is below 0.35 (substantially worse than global performance — suggests a domain gap that may require Australian-specific training data or a different pre-training base). The model systematically misses burns in dense eucalypt forest (possible that SAR signal differs from UK/Ireland vegetation types — consider whether additional continual pre-training on Australian scenes is needed).
