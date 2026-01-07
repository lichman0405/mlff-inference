# MatterSim Inference Detailed Installation Guide

## System Requirements

- Python 3.9-3.12
- CUDA 11.8+ (for GPU version)
- 8GB+ RAM (recommended 16GB+)
- 5GB+ disk space

## Installation Methods

### Method 1: Pip Installation (Recommended)

```bash
# CPU version
pip install mattersim-inference

# GPU version (requires CUDA pre-installed)
pip install mattersim-inference[gpu]
```

### Method 2: Conda Environment

```bash
# Create environment
conda create -n mattersim python=3.10
conda activate mattersim

# Install PyTorch (GPU)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# Install mattersim-inference
pip install mattersim-inference
```

### Method 3: Install from Source

```bash
git clone https://github.com/materials-ml/mattersim-inference
cd mattersim-inference
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Check version
python -c "import mattersim_inference; print(mattersim_inference.__version__)"

# Run test
python -c "from mattersim_inference import MatterSimInference; print('OK')"

# CLI test
mattersim-infer --help
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
from mattersim_inference import MatterSimInference

# Use GPU 0
calc = MatterSimInference(device="cuda:0")

# Use CPU
calc = MatterSimInference(device="cpu")
```

## Common Issues

### 1. MatterSim Installation Failed

```bash
# Try direct installation
pip install mattersim
```

### 2. CUDA Out of Memory

```python
# Use smaller model
calc = MatterSimInference(model_name="MatterSim-v1-1M")

# Or use CPU
calc = MatterSimInference(device="cpu")
```

### 3. Module Not Found

```bash
# Reinstall dependencies
pip install --upgrade mattersim ase phonopy
```

## Dependency Description

| Dependency | Version | Purpose |
|------|------|------|
| mattersim | >=1.0.0 | MatterSim model |
| ase | >=3.22.0 | Atomic simulation environment |
| phonopy | >=2.20.0 | Phonon calculations |
| torch | >=2.0.0 | Deep learning framework |
