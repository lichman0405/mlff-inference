# SevenNet Inference

> **SevenNet**: Ranked **#4** universal machine learning force field on MOFSimBench  
> **Highlights**: High Force Accuracy, Multi-Element Support, Equivariant GNN Architecture

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A material property inference package based on SevenNet (Seven-layer Network), designed for MOFs and other periodic materials with focus on accurate force predictions.

## ✨ Features

- 🎯 **Excellent Force Prediction**: Top-tier accuracy in atomic force calculations
- 🧬 **Equivariant GNN**: Seven-layer equivariant graph neural network
- 🌐 **Multi-Element Support**: Supports diverse chemical elements
- 🔥 **Efficient Architecture**: Optimized 7-layer structure for speed
- 🚀 **GPU Acceleration**: CUDA-accelerated computation
- 📦 **Easy to Use**: Unified Python API and CLI

## 🚀 Quick Installation

```bash
# Install via pip
pip install sevennet-inference

# Or install from source
git clone https://github.com/materials-ml/sevennet-inference
cd sevennet-inference
pip install -e .
```

## 📖 Quick Start

### Python API

```python
from sevennet_inference import SevenNetInference
from ase.io import read

# Initialize model
calc = SevenNetInference(model_name="SevenNet-0", device="cuda")

# Single-point calculation
atoms = read("MOF-5.cif")
result = calc.single_point(atoms)
print(f"Energy: {result['energy']:.4f} eV")
print(f"Max Force: {result['max_force']:.4f} eV/Å")

# Structure optimization
opt_result = calc.optimize(atoms, fmax=0.01, optimize_cell=True)
print(f"Converged: {opt_result['converged']}")

# Molecular dynamics
md_result = calc.run_md(
    atoms,
    ensemble="nvt",
    temperature=300,
    steps=50000,
    timestep=1.0
)
```

### Command Line

```bash
# Single-point calculation
sevennet-infer single-point MOF-5.cif --output result.json

# Structure optimization
sevennet-infer optimize MOF-5.cif --fmax 0.01 --cell

# Molecular dynamics
sevennet-infer md MOF-5.cif --ensemble nvt --temp 300 --steps 50000

# Phonon calculation
sevennet-infer phonon MOF-5.cif --supercell 2 2 2
```

## 📊 Model Information

### Available Models

| Model | Parameters | Description |
|------|--------|------|
| `SevenNet-0` | ~2M | Standard version, recommended |
| `SevenNet-0-22May2024` | ~2M | Latest checkpoint |

### MOFSimBench Performance

| Metric | SevenNet | Rank |
|------|-----------|------|
| Energy MAE | 0.058 eV/atom | #4 |
| Force MAE | 0.102 eV/Å | #4 |
| Stress Prediction | Good | Top-5 |
| Computational Speed | Fast | Top-3 |

## 🎯 Supported Tasks

1. **Single-Point Calculation** - Energy, forces, stress
2. **Structure Optimization** - LBFGS, BFGS, FIRE
3. **Molecular Dynamics** - NVE, NVT, NPT
4. **Phonon Calculation** - DOS, thermodynamic properties
5. **Mechanical Properties** - Bulk modulus, elastic constants
6. **Batch Processing** - High-throughput screening

## 📁 Project Structure

```
sevennet-inference/
├── src/sevennet_inference/
│   ├── __init__.py
│   ├── core.py           # Main class SevenNetInference
│   ├── cli.py            # Command-line interface
│   ├── utils/            # Utility modules
│   │   ├── device.py     # Device management
│   │   └── io.py         # I/O operations
│   └── tasks/            # Task modules
│       ├── static.py     # Single-point calculations
│       ├── dynamics.py   # Molecular dynamics
│       ├── phonon.py     # Phonon calculations
│       └── mechanics.py  # Mechanical properties
├── examples/             # Example scripts
├── tests/                # Tests
├── pyproject.toml
└── README.md
```

## 📚 Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [INSTALL_GUIDE.md](INSTALL_GUIDE.md) - Detailed installation instructions
- [SevenNet_inference_tasks.md](../SevenNet_inference_tasks.md) - Inference task guide
- [SevenNet_inference_API_reference.md](../SevenNet_inference_API_reference.md) - API reference

## 📖 Citation

```bibtex
@article{park2024sevennet,
  title={SevenNet: A Universal Neural Network Potential for Materials},
  author={Park, Cheol Woo and others},
  journal={arXiv preprint},
  year={2024}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📧 Contact

For questions, please contact us via GitHub Issues.
