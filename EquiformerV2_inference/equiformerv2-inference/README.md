# EquiformerV2 Inference

> **EquiformerV2**: Ranked **#5** universal machine learning force field on MOFSimBench  
> **Highlights**: Advanced Equivariance, High Scalability, Open Catalyst Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A material property inference package based on EquiformerV2, the next-generation equivariant transformer for molecular modeling, designed for MOFs and catalytic materials.

## ✨ Features

- 🎯 **Advanced Equivariance**: SO(3)-equivariant architecture with improved efficiency
- 🔬 **Transformer Architecture**: Attention-based graph neural network
- 🧬 **E(3) Symmetry**: Full rotational and translational equivariance
- 🌐 **Open Catalyst Project**: Trained on massive catalysis dataset
- 🚀 **GPU Optimized**: Efficient implementation for large-scale calculations
- 📦 **Easy to Use**: Unified Python API and CLI

## 🚀 Quick Installation

```bash
# Install via pip
pip install equiformerv2-inference

# Or install from source
git clone https://github.com/materials-ml/equiformerv2-inference
cd equiformerv2-inference
pip install -e .
```

## 📖 Quick Start

### Python API

```python
from equiformerv2_inference import EquiformerV2Inference
from ase.io import read

# Initialize model
calc = EquiformerV2Inference(model_name="EquiformerV2-31M-S2EF", device="cuda")

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
equiformerv2-infer single-point MOF-5.cif --output result.json

# Structure optimization
equiformerv2-infer optimize MOF-5.cif --fmax 0.01 --cell

# Molecular dynamics
equiformerv2-infer md MOF-5.cif --ensemble nvt --temp 300 --steps 50000

# Phonon calculation
equiformerv2-infer phonon MOF-5.cif --supercell 2 2 2
```

## 📊 Model Information

### Available Models

| Model | Parameters | Description |
|------|--------|------|
| `EquiformerV2-31M-S2EF` | 31M | Structure-to-Energy-and-Forces |
| `EquiformerV2-153M-S2EF` | 153M | Large model for high accuracy |

### MOFSimBench Performance

| Metric | EquiformerV2 | Rank |
|------|-----------|------|
| Energy MAE | 0.062 eV/atom | #5 |
| Force MAE | 0.108 eV/Å | #5 |
| Computational Efficiency | Good | Top-5 |
| Scalability | Excellent | Top-3 |

## 🎯 Supported Tasks

1. **Single-Point Calculation** - Energy, forces, stress
2. **Structure Optimization** - LBFGS, BFGS, FIRE
3. **Molecular Dynamics** - NVE, NVT, NPT
4. **Phonon Calculation** - DOS, thermodynamic properties
5. **Mechanical Properties** - Bulk modulus, elastic constants
6. **Batch Processing** - High-throughput screening

## 📁 Project Structure

```
equiformerv2-inference/
├── src/equiformerv2_inference/
│   ├── __init__.py
│   ├── core.py           # Main class EquiformerV2Inference
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
- [EquiformerV2_inference_tasks.md](../EquiformerV2_inference_tasks.md) - Inference task guide
- [EquiformerV2_inference_API_reference.md](../EquiformerV2_inference_API_reference.md) - API reference

## 📖 Citation

```bibtex
@article{liao2023equiformerv2,
  title={EquiformerV2: Improved Equivariant Transformer for Scalable and Accurate Interatomic Potentials},
  author={Liao, Yi-Lun and Smidt, Tess},
  journal={arXiv preprint arXiv:2306.12059},
  year={2023}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📧 Contact

For questions, please contact us via GitHub Issues.
