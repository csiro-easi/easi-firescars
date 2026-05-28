#!/usr/bin/env bash
# Install additional dependencies into the existing burn-scar venv.
# Assumes ~/venvs/burn-scar already exists per EASI venv-setup.md.
set -e

source ~/venvs/burn-scar/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install terratorch impactmesh huggingface_hub rasterio tensorboard

echo "=== Downloading ImpactMesh-Fire (S1, S2, MASK only — ~30 GB) ==="
hf download ibm-esa-geospatial/ImpactMesh-Fire \
  --repo-type dataset \
  --include "train/*.tar" \
  --include "val/*.tar" \
  --include "test/*.tar" \
  --include "split/*" \
  --exclude "*DEM*" \
  --local-dir data/ImpactMesh-Fire

echo "=== Extracting ==="
mkdir -p data/ImpactMesh-Fire/data
for f in data/ImpactMesh-Fire/*/*.tar; do
  echo "Extracting $f"
  tar -xf "$f" -C data/ImpactMesh-Fire/data
done

echo "=== Removing tar files ==="
rm -f data/ImpactMesh-Fire/train/*.tar data/ImpactMesh-Fire/val/*.tar data/ImpactMesh-Fire/test/*.tar

echo "=== Done ==="
echo "Run: python -m pytest tests/ -k 'not test_model'"
echo "Then open workflow.ipynb with the burn-scar kernel."
