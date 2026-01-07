# SevenNet Inference Detailed Installation Guide

## System Requirements

- Python 3.9-3.11
- CUDA 11.8+ (for GPU version)
- 8GB+ RAM (recommended 16GB+)
- 5GB+ disk space

## Installation Methods

### Method 1: Pip Installation (Recommended)

```bash
# CPU version
pip install sevennet-inference

# GPU version (requires CUDA pre-installed)
pip install sevennet-inference[gpu]
```

### Method 2: Conda Environment

```bash
# Create environment
conda create -n sevennet python=3.10
conda activate sevennet

# Install PyTorch (GPU)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install sevennet-inference
pip install sevennet-inference
```

### Method 3: Install from Source

```bash
git clone https://github.com/materials-ml/sevennet-inference
cd sevennet-inference
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Check version
python -c "import sevennet_inference; print(sevennet_inference.__version__)"

# Run test
python -c "from sevennet_inference import SevenNetInference; print('OK')"

# CLI test
sevennet-infer --help
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
from sevennet_inference import SevenNetInference

# Use GPU 0
calc = SevenNetInference(device="cuda:0")

# Use CPU
calc = SevenNetInference(device="cpu")
```

## Common Issues

### 1. SevenNet Installation Failed

```bash
# Try direct installation
pip install sevenn

# Or from GitHub
pip install git+https://github.com/MDIL-SNU/SevenNet.git
```

### 2. CUDA Out of Memory

```python
# Use smaller batch size
calc = SevenNetInference(batch_size=16)

# Or use CPU
calc = SevenNetInference(device="cpu")
```

### 3. Module Not Found

```bash
# Reinstall dependencies
pip install --upgrade sevenn ase phonopy
```

## Dependency Description

| Dependency | Version | Purpose |
|------|------|------|
| sevenn | >=0.9.0 | SevenNet model |
| ase | >=3.22.0 | Atomic simulation environment |
| phonopy | >=2.20.0 | Phonon calculations |
| torch | >=2.0.0 | Deep learning framework |
