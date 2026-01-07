# Installation Guide - Orb Inference

Complete installation guide for orb-inference on CPU and GPU systems.

## System Requirements

### Hardware

- **CPU**: Any modern x86_64 processor
- **GPU** (optional): NVIDIA GPU with CUDA 11.8+ or 12.1+ support
- **RAM**: 8 GB minimum, 16 GB+ recommended for large systems
- **Storage**: ~5 GB for models and dependencies

### Software

- **OS**: Linux, macOS, Windows (WSL2 recommended for Windows)
- **Python**: 3.8, 3.9, 3.10, or 3.11 (3.10 recommended)
  - **Not supported**: Python 3.12+ (PyTorch compatibility issues)
  - **Not recommended**: Python 3.7 and below

## Quick Installation

### Option 1: CPU-Only (Simplest)

```bash
# Clone repository
git clone https://github.com/MLFF-inference/orb-inference
cd orb-inference

# Install dependencies
pip install -r requirements-cpu.txt

# Install package in editable mode
pip install -e .

# Verify installation
python tests/test_install.py
```

### Option 2: GPU (CUDA 12.1)

```bash
# Clone repository
git clone https://github.com/MLFF-inference/orb-inference
cd orb-inference

# Install PyTorch with CUDA 12.1
pip install -r requirements-gpu.txt

# Install package
pip install -e .

# Verify installation
python tests/test_install.py
```

## Detailed Installation

### Step 1: Python Environment

We recommend using conda or venv to create an isolated environment:

#### Using Conda (Recommended)

```bash
# Create conda environment
conda create -n orb-env python=3.10
conda activate orb-env

# Verify Python version
python --version  # Should show Python 3.10.x
```

#### Using venv

```bash
# Create virtual environment
python3.10 -m venv orb-env
source orb-env/bin/activate  # On Windows: orb-env\Scripts\activate

# Verify Python version
python --version
```

### Step 2: Install PyTorch

**CRITICAL**: Use PyTorch **2.3.1** (avoid 2.4.1 due to compatibility issues)

#### CPU-Only

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

#### GPU (CUDA 11.8)

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

#### GPU (CUDA 12.1)

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

#### Verify PyTorch Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Step 3: Install Core Dependencies

```bash
# ASE (Atomic Simulation Environment)
pip install ase>=3.23.0

# Phonopy
pip install phonopy>=2.20.0

# Orb models (auto-installs dependencies)
pip install orb-models

# Additional dependencies
pip install numpy>=1.20.0 scipy>=1.7.0 matplotlib>=3.5.0 click>=8.0.0
```

### Step 4: Install Orb Inference

```bash
# Navigate to repository
cd orb-inference

# Install in editable mode
pip install -e .

# Or install from requirements file
pip install -r requirements-cpu.txt  # or requirements-gpu.txt
pip install -e .
```

### Step 5: Verify Installation

Run the installation test script:

```bash
python tests/test_install.py
```

**Expected output:**

```
============================================================
orb-inference Installation Test
============================================================
Checking Python version...
  Python 3.10.12
  ✓ Python version OK

Checking PyTorch...
  PyTorch version: 2.3.1
  CUDA available: Yes
  CUDA version: 12.1
  GPU: NVIDIA RTX 4090
  ✓ PyTorch OK

Checking ASE...
  ASE version: 3.23.0
  ✓ ASE OK

Checking orb-models...
  orb-models installed
  ✓ orb-models OK

...

============================================================
✓ All critical tests passed!
Installation is ready for use.
============================================================
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Install build tools (for some packages)
sudo apt install build-essential

# Continue with standard installation
```

### macOS

```bash
# Install Python 3.10 via Homebrew
brew install python@3.10

# For Apple Silicon (M1/M2), use MPS acceleration
pip install torch==2.3.1

# Continue with standard installation
```

### Windows (WSL2 Recommended)

**Option 1: WSL2 (Recommended)**

```bash
# Install WSL2 with Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL, follow Linux instructions
sudo apt update
sudo apt install python3.10 python3.10-venv
```

**Option 2: Native Windows**

```powershell
# Install Python 3.10 from python.org
# Use PowerShell or Command Prompt

# Create virtual environment
python -m venv orb-env
orb-env\Scripts\activate

# Continue with standard installation
```

## Troubleshooting

### Issue 1: PyTorch 2.4.1 Compatibility

**Symptoms**: Import errors, CUDA errors, or model loading failures

**Solution**: Downgrade to PyTorch 2.3.1

```bash
pip uninstall torch
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: ASE FrechetCellFilter Error

**Symptoms**: `ImportError: cannot import name 'FrechetCellFilter'`

**Solution**: Upgrade ASE to 3.23.0+

```bash
pip install --upgrade ase>=3.23.0
```

### Issue 3: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. Reduce precision:
   ```python
   orb = OrbInference(model_name='orb-v3-omat', precision='float32-high')
   ```

2. Use CPU for large systems:
   ```python
   orb = OrbInference(model_name='orb-v3-omat', device='cpu')
   ```

3. Reduce system size or supercell dimensions

### Issue 4: Phonopy ImportError

**Symptoms**: `ModuleNotFoundError: No module named 'phonopy'`

**Solution**:

```bash
pip install phonopy>=2.20.0
```

### Issue 5: orb-models Installation Failure

**Symptoms**: `ERROR: Could not find a version that satisfies the requirement orb-models`

**Solution**: Install from GitHub or PyPI directly

```bash
# Try PyPI
pip install orb-models

# If fails, install from GitHub
pip install git+https://github.com/orbital-materials/orb-models.git
```

### Issue 6: Model Download Timeout

**Symptoms**: Model download hangs or times out

**Solution**: Models are downloaded on first use. Ensure stable internet connection.

```python
# Pre-download model
from orb_inference import OrbInference
orb = OrbInference(model_name='orb-v3-omat', device='cpu')
# Model is now cached locally
```

### Issue 7: Permission Errors (Linux/macOS)

**Symptoms**: `PermissionError` during installation

**Solution**: Use `--user` flag or virtual environment

```bash
# Option 1: User installation
pip install --user -e .

# Option 2: Virtual environment (recommended)
python -m venv orb-env
source orb-env/bin/activate
pip install -e .
```

## Testing Your Installation

### Quick Test

```python
from orb_inference import OrbInference
from ase.build import bulk

# Create test structure
atoms = bulk('Cu', 'fcc', a=3.6)

# Initialize model
orb = OrbInference(model_name='orb-v3-omat', device='cuda')

# Calculate energy
result = orb.single_point(atoms)
print(f"Energy: {result['energy']:.4f} eV")
# Expected: Energy ~ -3.5 eV (model-dependent)
```

### Run All Examples

```bash
cd examples
python 01_basic_usage.py
python 02_molecular_dynamics.py
python 03_phonon_calculation.py
python 04_mechanical_properties.py
python 05_high_throughput.py
```

## Advanced Configuration

### Proxy Settings

If behind a corporate firewall:

```bash
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
pip install -e .
```

### Offline Installation

1. Download dependencies on a machine with internet:
   ```bash
   pip download -r requirements-cpu.txt -d packages/
   ```

2. Transfer `packages/` to offline machine

3. Install offline:
   ```bash
   pip install --no-index --find-links=packages/ -r requirements-cpu.txt
   pip install -e .
   ```

### Custom CUDA Paths

If CUDA is installed in non-standard location:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Uninstallation

```bash
# Uninstall orb-inference
pip uninstall orb-inference

# Remove environment
conda deactivate
conda env remove -n orb-env

# Or with venv
deactivate
rm -rf orb-env
```

## Getting Help

- **Installation Issues**: [GitHub Issues](https://github.com/MLFF-inference/orb-inference/issues)
- **Documentation**: [README.md](README.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)

## FAQ

**Q: Which Python version should I use?**

A: Python 3.10 is recommended. Versions 3.8-3.11 are supported. Avoid 3.12+.

**Q: Do I need a GPU?**

A: No, CPU-only mode works fine. GPU significantly speeds up calculations for large systems.

**Q: Which Orb model should I use?**

A: Start with `orb-v3-omat` (OMAT24 dataset, conservative forces). See [model comparison](README.md#orb-models).

**Q: How much disk space do I need?**

A: ~5 GB for models and dependencies. Models are cached locally after first download.

**Q: Can I use on Windows?**

A: Yes, but WSL2 is strongly recommended for better compatibility.

**Q: How do I update orb-inference?**

A:
```bash
cd orb-inference
git pull
pip install -e . --upgrade
```
