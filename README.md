# Burn Scar Fine-Tuning with granite-geospatial-uki

Fine-tune the IBM granite-geospatial-uki foundation model (8-band: Sentinel-1 + Sentinel-2) for binary burn scar segmentation, evaluated on Australian Black Summer fire events.

## Structure

```
├── science-plan.md          # Research objectives and validation criteria
├── engineering-plan.md      # Technical implementation specification
├── workflow.ipynb           # Orchestration notebook (run this)
├── modules/
│   ├── data.py              # Dataset, normalisation, split loading
│   ├── model.py             # GraniteUKIBurnScar model
│   ├── train.py             # Training loop, loss, metrics
│   └── visualisation.py     # Plots and prediction panels
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_train.py
├── setup.sh                 # Install deps + download data
└── easi-notebooks-ddp/      # Reference DDP code (PR #32)
```

## Installation

```bash
# Assumes ~/venvs/burn-scar exists
source ~/venvs/burn-scar/bin/activate
bash setup.sh
```

## Running Tests

```bash
source ~/venvs/burn-scar/bin/activate
python -m pytest tests/ -k "not test_model"  # skip model tests if terratorch unavailable
python -m pytest tests/                       # full suite (requires terratorch + GPU)
```

## Usage

Open `workflow.ipynb` with the "Burn Scar Fine-Tuning" Jupyter kernel and run cells sequentially.

## Data Requirements

- ImpactMesh-Fire dataset (~30 GB) at `data/ImpactMesh-Fire/`
- GPU with 24+ GB VRAM (A10, V100, or A100)

## Key Details

- **Input**: 8 bands (B02, B03, B04, B8A, B11, B12, VV, VH) at 224×224 px
- **Model**: ~100M param ViT backbone + convolutional decoder
- **Training**: 2-phase (frozen backbone → full fine-tuning)
- **S1 normalisation**: 10×log₁₀(σ₀), clip [-35, 10], scale to [0, 1]
- **S2 normalisation**: divide by 10000, clip [0, 1]
