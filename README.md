# Fire Scar Detection with Prithvi-EO-2.0

Fine-tunes the NASA/IBM Prithvi-EO-2.0-300M geospatial foundation model for binary burned area segmentation using the HLS Burn Scars dataset on EASI.

## Structure

```
fire-scar-prithvi/
├── workflow.ipynb           # Orchestration notebook (run this)
├── burn_scars_config.yaml  # TerraTorch training configuration
├── conftest.py             # Test discovery support
├── modules/
│   ├── data_access.py      # Dataset download and verification
│   ├── training.py         # Config generation, GPU check, training
│   ├── evaluation.py       # IoU, precision, recall metrics
│   ├── inference.py        # Model loading, prediction, GeoTIFF export
│   └── visualisation.py    # False-colour, curves, comparison plots
└── tests/
    ├── test_data_access.py
    ├── test_evaluation.py
    ├── test_inference.py
    ├── test_training.py
    └── test_visualisation.py
```

## Installation

```bash
pip install terratorch huggingface_hub rasterio matplotlib pyyaml
```

Requires EASI CUDA container image for GPU training.

## Running Tests

```bash
cd fire-scar-prithvi
pytest tests/ -v
```

## Usage

1. Spawn a GPU server on EASI (CUDA image, g4dn instance, 1 GPU)
2. Open `workflow.ipynb`
3. Run all cells — defaults to quick-test mode (~10 min)
4. Set `QUICK_TEST = False` for full benchmark reproduction (~2-4 hours)

## Data Requirements

- **HLS Burn Scars dataset**: Downloaded automatically from Hugging Face
- **Prithvi-EO-2.0-300M weights**: Downloaded automatically by TerraTorch
- No manual data preparation needed

## Key Details

- **Bands:** Blue, Green, Red, NIR Narrow, SWIR1, SWIR2 (6 bands at 30m)
- **Architecture:** Prithvi-EO-2.0-300M backbone + UNet decoder
- **Benchmark:** 87.5% IoU (burned class), 93.0% mean IoU
- **Quick test:** 80 chips, 5 epochs, ~10 min, confirms pipeline works
