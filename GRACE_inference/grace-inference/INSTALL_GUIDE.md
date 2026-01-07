# GRACE Inference - Complete Installation and Usage Guide

## 📦 Project Structure

```
grace-inference/
├── src/grace_inference/          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── core.py                  # GRACEInference main class
│   ├── cli.py                   # Command-line interface
│   ├── tasks/                   # Task modules
│   │   ├── __init__.py
│   │   ├── static.py            # Single-point energy, structure optimization
│   │   ├── dynamics.py          # Molecular dynamics
│   │   ├── phonon.py            # Phonon calculations
│   │   ├── mechanics.py         # Mechanical properties
│   │   └── adsorption.py        # Adsorption energy, coordination analysis
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── device.py            # Device management (CPU/GPU)
│       ├── graph.py             # Graph construction for DGL
│       └── io.py                # Structure I/O
├── examples/                    # Usage examples
│   ├── 01_single_point.py
│   ├── 02_structure_optimization.py
│   ├── 03_molecular_dynamics.py
│   ├── 04_phonon_calculation.py
│   ├── 05_bulk_modulus.py
│   ├── 06_adsorption_energy.py
│   ├── 07_coordination_analysis.py
│   └── 08_high_throughput.py
├── tests/                       # Unit tests
│   ├── test_install.py
│   └── test_utils.py
├── pyproject.toml               # Project configuration
├── setup.py                     # Setup script (backward compatibility)
├── README.md                    # Project description
├── QUICKSTART.md                # Quick start guide
├── INSTALL_GUIDE.md             # This file
├── LICENSE                      # MIT License
└── .gitignore                   # Git ignore file
```

## 🚀 Installation Steps

### Prerequisites

- **Python**: 3.9, 3.10, or 3.11 (recommended: 3.10)
- **Operating System**: Windows, Linux, or macOS
- **Hardware**: CPU or NVIDIA GPU with CUDA support

### Method 1: Local Development Installation (Recommended)

#### Step 1: Clone or Navigate to Repository

```bash
cd c:\Users\lishi\code\MLFF-inference\GRACE_inference\grace-inference
```

#### Step 2: Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n grace-inference-cpu python=3.10
conda activate grace-inference-cpu
```

#### Step 3: Install PyTorch (CPU Version)

```bash
# For CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8 (GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1 (GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Step 4: Install DGL

```bash
# For CPU
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# For CUDA 11.8 (GPU)
# pip install dgl-cu118 -f https://data.dgl.ai/wheels/cu118/repo.html

# For CUDA 12.1 (GPU)
# pip install dgl-cu121 -f https://data.dgl.ai/wheels/cu121/repo.html
```

#### Step 5: Install GRACE Inference

```bash
# Basic installation (CPU version)
pip install -e .

# Or install with all features
pip install -e ".[all]"
```

### Method 2: Modular Installation

```bash
# CPU version only
pip install -e .

# GPU support (requires CUDA)
pip install -e ".[gpu]"

# Development version (includes testing and linting tools)
pip install -e ".[dev]"

# All features
pip install -e ".[all]"
```

### Method 3: Using Requirements Files

```bash
# Navigate to GRACE_inference directory
cd c:\Users\lishi\code\MLFF-inference\GRACE_inference

# CPU installation
conda create -n grace-inference-cpu python=3.10
conda activate grace-inference-cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install dgl -f https://data.dgl.ai/wheels/repo.html
pip install -r requirements-cpu.txt
cd grace-inference
pip install -e .

# GPU installation (CUDA 11.8)
conda create -n grace-inference-gpu python=3.10
conda activate grace-inference-gpu
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install dgl-cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
pip install -r requirements-gpu.txt
cd grace-inference
pip install -e .
```

### Post-Installation Verification

```bash
# Test Python import
python -c "from grace_inference import GRACEInference; print('✓ GRACE Inference installed successfully!')"

# Check version
python -c "import grace_inference; print(f'Version: {grace_inference.__version__}')"

# Test CLI
grace-infer --version
grace-infer info

# Run installation test
python tests/test_install.py
```

## 📚 Python API Usage

### 1. Basic Setup

```python
from grace_inference import GRACEInference

# Initialize with default settings
calc = GRACEInference()

# Initialize with specific model
calc = GRACEInference(model="mof-v1", device="auto")

# Initialize with custom checkpoint
calc = GRACEInference(model_path="/path/to/checkpoint.pt", device="cuda")
```

### 2. Single-Point Energy Calculation

```python
from ase.io import read

# Load structure
atoms = read("mof.cif")

# Calculate energy and forces
result = calc.single_point(atoms)

print(f"Energy: {result['energy']:.6f} eV")
print(f"Forces shape: {result['forces'].shape}")
print(f"Stress shape: {result['stress'].shape}")
```

### 3. Structure Optimization

```python
# Quick optimization
optimized = calc.optimize("mof.cif", fmax=0.05)

# Advanced optimization with custom settings
optimized = calc.optimize(
    structure="mof.cif",
    fmax=0.05,                    # Force convergence criterion (eV/Å)
    max_steps=500,                # Maximum steps
    optimizer="BFGS",             # BFGS, LBFGS, FIRE, etc.
    trajectory="opt.traj",        # Save optimization trajectory
    output="optimized.cif",       # Save final structure
    logfile="opt.log"             # Optimization log
)

# Check convergence
print(f"Converged: {optimized.info['converged']}")
print(f"Steps: {optimized.info['steps']}")
print(f"Final energy: {optimized.get_potential_energy():.6f} eV")
```

### 4. Molecular Dynamics

```python
# NVT ensemble
final_atoms = calc.run_md(
    structure="mof.cif",
    ensemble="nvt",               # NVT, NVE, NPT
    temperature_K=300,            # Temperature (K)
    steps=10000,                  # Number of steps
    timestep_fs=1.0,              # Timestep (fs)
    trajectory="md.traj",         # Save trajectory
    log_interval=100,             # Log every N steps
    traj_interval=10              # Save trajectory every N steps
)

# NPT ensemble with pressure
final_atoms = calc.run_md(
    structure="mof.cif",
    ensemble="npt",
    temperature_K=300,
    pressure_bar=1.0,             # Pressure (bar)
    steps=50000,
    timestep_fs=0.5,
    trajectory="npt.traj"
)
```

### 5. Phonon Calculations

```python
# Calculate phonon properties
result = calc.phonon(
    structure="mof.cif",
    supercell_matrix=[2, 2, 2],   # Supercell size
    temp_range=[0, 1000, 10],     # Temperature range (K)
    save_band_yaml="band.yaml",   # Band structure
    save_dos_dat="dos.dat",       # Density of states
    mesh=[50, 50, 50]             # q-point mesh
)

# Access thermodynamic properties
print(f"Zero-point energy: {result['ZPE']:.4f} eV")
print(f"Heat capacity (300K): {result['Cv_300K']:.4f} J/mol/K")
print(f"Entropy (300K): {result['S_300K']:.4f} J/mol/K")
print(f"Free energy (300K): {result['F_300K']:.4f} eV")
```

### 6. Bulk Modulus

```python
# Calculate bulk modulus
result = calc.bulk_modulus(
    structure="mof.cif",
    strain_range=0.05,            # ±5% strain
    n_points=7,                   # Number of data points
    fit_order=3                   # Polynomial fit order
)

print(f"Bulk modulus: {result['B_GPa']:.2f} GPa")
print(f"Equilibrium volume: {result['V0']:.3f} Å³")
print(f"Equilibrium energy: {result['E0']:.6f} eV")
```

### 7. Gas Adsorption Energy

```python
# Calculate adsorption energy
result = calc.adsorption_energy(
    mof_structure="IRMOF-1.cif",
    gas_molecule="CO2",                  # CO2, H2, CH4, N2, etc.
    site_position=[10.0, 10.0, 10.0],   # Adsorption site (Å)
    relax_mof=True,                     # Relax MOF structure
    relax_complex=True,                 # Relax MOF+gas complex
    fmax=0.05                           # Force convergence
)

print(f"Adsorption energy: {result['E_ads']:.3f} eV")
print(f"Distance: {result['distance']:.3f} Å")
print(f"E_mof: {result['E_mof']:.6f} eV")
print(f"E_gas: {result['E_gas']:.6f} eV")
print(f"E_complex: {result['E_complex']:.6f} eV")
```

### 8. Coordination Analysis

```python
# Analyze metal coordination environment
result = calc.analyze_coordination(
    structure="MOF-5.cif",
    center_element="Zn",          # Metal center
    ligand_element="O",           # Coordinating atom
    cutoff_radius=3.0,            # Cutoff distance (Å)
    angle_cutoff=30.0             # Angle threshold (degrees)
)

print(f"Number of centers: {result['n_centers']}")
print(f"Average coordination: {result['avg_coordination']:.2f}")
print(f"Coordination distribution: {result['coordination_distribution']}")
print(f"Average distance: {result['avg_distance']:.3f} Å")
```

## 🔧 Command Line Interface Usage

### Basic Commands

```bash
# Show help
grace-infer --help

# Show version
grace-infer --version

# Show system info
grace-infer info
grace-infer info --verbose
```

### Single-Point Energy

```bash
grace-infer energy mof.cif
grace-infer energy mof.cif --model mof-v1
grace-infer energy mof.cif --device cuda
grace-infer energy mof.cif --output result.json
```

### Structure Optimization

```bash
grace-infer optimize mof.cif
grace-infer optimize mof.cif --fmax 0.05
grace-infer optimize mof.cif --max-steps 500
grace-infer optimize mof.cif --optimizer BFGS
grace-infer optimize mof.cif --output optimized.cif
grace-infer optimize mof.cif --trajectory opt.traj
```

### Molecular Dynamics

```bash
grace-infer md mof.cif --ensemble nvt --temp 300 --steps 10000
grace-infer md mof.cif --ensemble npt --temp 300 --pressure 1.0 --steps 50000
grace-infer md mof.cif --timestep 1.0 --trajectory md.traj
grace-infer md mof.cif --log-interval 100 --traj-interval 10
```

### Phonon Calculation

```bash
grace-infer phonon mof.cif --supercell 2 2 2
grace-infer phonon mof.cif --supercell 2 2 2 --temp-range 0 1000 10
grace-infer phonon mof.cif --mesh 50 50 50
grace-infer phonon mof.cif --save-band band.yaml --save-dos dos.dat
```

### Bulk Modulus

```bash
grace-infer bulk-modulus mof.cif
grace-infer bulk-modulus mof.cif --strain 0.05
grace-infer bulk-modulus mof.cif --n-points 7
grace-infer bulk-modulus mof.cif --output bm_result.json
```

### Adsorption Energy

```bash
grace-infer adsorption mof.cif --gas CO2 --site 10.0 10.0 10.0
grace-infer adsorption mof.cif --gas H2 --site auto
grace-infer adsorption mof.cif --gas CH4 --site 10 10 10 --relax
grace-infer adsorption mof.cif --gas N2 --site 10 10 10 --fmax 0.05
```

### Coordination Analysis

```bash
grace-infer coordination mof.cif --center Zn
grace-infer coordination mof.cif --center Zn --ligand O
grace-infer coordination mof.cif --center Zn --cutoff 3.0
grace-infer coordination mof.cif --center Zn --output coord.json
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/
```

### Run Specific Tests

```bash
# Installation test
pytest tests/test_install.py

# Utility tests
pytest tests/test_utils.py

# With coverage report
pytest --cov=grace_inference tests/

# Verbose output
pytest -v tests/
```

### Quick Installation Check

```bash
python tests/test_install.py
```

## 🐛 Troubleshooting

### Issue 1: Import Error

**Problem**: `ModuleNotFoundError: No module named 'grace_inference'`

**Solution**:
```bash
# Check installation
pip list | grep grace

# Reinstall
pip install -e . --force-reinstall

# Verify
python -c "import grace_inference; print('OK')"
```

### Issue 2: CUDA Not Available

**Problem**: GPU not detected or CUDA errors

**Solution**:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check DGL CUDA support
python -c "import dgl; print(f'DGL CUDA: {dgl.cuda.is_available()}')"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Reinstall DGL with matching CUDA version
pip uninstall dgl
pip install dgl-cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
```

### Issue 3: DGL Installation Issues

**Problem**: DGL installation fails or version mismatch

**Solution**:
```bash
# CPU version
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# GPU version (CUDA 11.8)
pip install dgl-cu118 -f https://data.dgl.ai/wheels/cu118/repo.html

# Verify installation
python -c "import dgl; print(dgl.__version__)"
```

### Issue 4: Memory Issues

**Problem**: Out of memory errors on GPU

**Solution**:
```python
# Reduce batch size
calc = GRACEInference(batch_size=1)

# Use CPU instead
calc = GRACEInference(device="cpu")

# Clear CUDA cache
import torch
torch.cuda.empty_cache()
```

### Issue 5: Phonopy Not Found

**Problem**: `ImportError: No module named 'phonopy'`

**Solution**:
```bash
pip install phonopy>=2.20.0
```

### Issue 6: ASE Structure Issues

**Problem**: Cannot read CIF files

**Solution**:
```bash
# Install ASE with CIF support
pip install ase>=3.22.0

# Test reading
python -c "from ase.io import read; atoms = read('test.cif'); print(atoms)"
```

## 📊 Performance Optimization

### GPU Optimization

```python
# Use mixed precision (faster inference)
calc = GRACEInference(device="cuda", precision="float16")

# Increase batch size for parallel processing
calc = GRACEInference(batch_size=32)

# Use optimized DGL backend
import dgl
dgl.use_libxsmm(True)
```

### CPU Optimization

```python
# Use multi-threading
import torch
torch.set_num_threads(8)

# Optimize for CPU
calc = GRACEInference(device="cpu", optimize_cpu=True)
```

## 🔍 Advanced Configuration

### Custom Model Loading

```python
# Load from checkpoint
calc = GRACEInference(
    model_path="/path/to/checkpoint.pt",
    device="cuda"
)

# Load with custom config
calc = GRACEInference(
    model="mof-v1",
    config={
        'cutoff': 6.0,
        'max_neighbors': 50,
        'hidden_dim': 128
    }
)
```

### Logging and Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Save detailed logs
calc = GRACEInference(log_file="grace_debug.log", verbose=True)
```

## 📖 Additional Resources

- **GitHub Repository**: [https://github.com/yourusername/grace-inference](https://github.com/yourusername/grace-inference)
- **Documentation**: [README.md](README.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Examples**: [examples/](examples/)
- **Issue Tracker**: [GitHub Issues](https://github.com/yourusername/grace-inference/issues)

## 🤝 Getting Help

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/yourusername/grace-inference/issues)
2. Search [Stack Overflow](https://stackoverflow.com/questions/tagged/grace-inference)
3. Post in [GitHub Discussions](https://github.com/yourusername/grace-inference/discussions)
4. Contact the development team

---

**Happy computing with GRACE Inference! 🚀**
