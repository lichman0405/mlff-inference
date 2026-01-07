# EquiformerV2 Inference Detailed Installation Guide

## System Requirements

- Python 3.9-3.11
- CUDA 11.8+ (for GPU version)
- 12GB+ RAM (recommended 16GB+)
- 8GB+ disk space

## Installation Methods

### Method 1: Pip Installation (Recommended)

```bash
# CPU version
pip install equiformerv2-inference

# GPU version (requires CUDA pre-installed)
pip install equiformerv2-inference[gpu]
```

### Method 2: Conda Environment

```bash
# Create environment
conda create -n equiformerv2 python=3.10
conda activate equiformerv2

# Install PyTorch (GPU)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install torch-geometric
pip install torch-geometric

# Install equiformerv2-inference
pip install equiformerv2-inference
```

### Method 3: Install from Source

```bash
git clone https://github.com/materials-ml/equiformerv2-inference
cd equiformerv2-inference
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Check version
python -c "import equiformerv2_inference; print(equiformerv2_inference.__version__)"

# Run test
python -c "from equiformerv2_inference import EquiformerV2Inference; print('OK')"

# CLI test
equiformerv2-infer --help
```

## GPU Configuration

### Check CUDA

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

### Use Specific GPU

```python
from equiformerv2_inference import EquiformerV2Inference

# Use GPU 0
calc = EquiformerV2Inference(device="cuda:0")

# Use CPU
calc = EquiformerV2Inference(device="cpu")
```

## Common Issues

### 1. E3NN Installation Failed

```bash
# Install e3nn separately
pip install e3nn

# Or from conda
conda install -c conda-forge e3nn
```

### 2. Torch-Geometric Issues

```bash
# Install torch-geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

### 3. CUDA Out of Memory

```python
# Use smaller model
calc = EquiformerV2Inference(model_name="EquiformerV2-31M-S2EF")

# Or use CPU
calc = EquiformerV2Inference(device="cpu")
```

### 4. Module Not Found

```bash
# Reinstall dependencies
pip install --upgrade e3nn torch-geometric ase phonopy
```

## Dependency Description

| Dependency | Version | Purpose |
|------|------|------|
| equiformer-v2 | >=0.1.0 | EquiformerV2 model |
| e3nn | >=0.5.0 | E(3) equivariant operations |
| torch-geometric | >=2.3.0 | Graph neural networks |
| ase | >=3.22.0 | Atomic simulation environment |
| phonopy | >=2.20.0 | Phonon calculations |
| torch | >=2.0.0 | Deep learning framework |
