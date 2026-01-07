# MatterSim Inference

> **MatterSim**: Ranked **#3** universal machine learning force field on MOFSimBench  
> **Highlights**: Adsorption Energy #1 🥇, MD Stability #1 🥇, Uncertainty Estimation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A material property inference package based on MatterSim, designed for MOFs (Metal-Organic Frameworks) and other periodic materials.

## ✨ Features

- 🎯 **#1 in Adsorption Energy**: Best host-guest interaction modeling
- 🔬 **#1 in MD Stability**: Tied with eSEN for best stability
- 📊 **Uncertainty Estimation**: Enabled via model ensemble
- 🔥 **Three-Body Interactions**: Precise angular dependence modeling
- 🚀 **GPU Acceleration**: CUDA-accelerated computation
- 📦 **Easy to Use**: Unified Python API and CLI

## 🚀 Quick Installation

```bash
# Install via pip
pip install mattersim-inference

# Or install from source
git clone https://github.com/lichman0405/mlff-inference.git
cd mlff-inference/MatterSim_inference/mattersim-inference
pip install -e .
```

## 📖 Quick Start

### Python API

```python
from mattersim_inference import MatterSimInference
from ase.io import read

# Initialize model
calc = MatterSimInference(model_name="MatterSim-v1-5M", device="cuda")

# Single-point calculation
atoms = read("MOF-5.cif")
result = calc.single_point(atoms)
print(f"Energy: {result['energy']:.4f} eV")

# Structure optimization
opt_result = calc.optimize(atoms, fmax=0.01, optimize_cell=True)
print(f"Converged: {opt_result['converged']}")

# Adsorption energy (MatterSim's strongest feature)
ads_result = calc.adsorption_energy(
    mof_structure=atoms,
    gas_molecule="CO2",
    site_position=[10.0, 10.0, 10.0]
)
print(f"Adsorption Energy: {ads_result['E_ads']:.4f} eV")
```

### Command Line

```bash
# Single-point calculation
mattersim-infer single-point MOF-5.cif --output result.json

# Structure optimization
mattersim-infer optimize MOF-5.cif --fmax 0.01 --cell

# Molecular dynamics
mattersim-infer md MOF-5.cif --ensemble nvt --temp 300 --steps 50000

# Adsorption energy
mattersim-infer adsorption MOF.cif --gas CO2 --site 10 10 10
```

## 📊 Model Information

### Available Models

| Model | Parameters | Description |
|------|--------|------|
| `MatterSim-v1-1M` | 1M | Lightweight, fast testing |
| `MatterSim-v1-5M` | 5M | **Recommended for production** |

### MOFSimBench Performance

| Metric | MatterSim | Rank |
|------|-----------|------|
| Energy MAE | 0.052 eV/atom | #3 |
| Force MAE | 0.095 eV/Å | #3 |
| **Adsorption Energy** | **Best** | **#1** 🥇 |
| **MD Stability** | **Excellent** | **#1** 🥇 |

## 🎯 Supported Tasks

1. **Single-Point Calculation** - Energy, forces, stress
2. **Structure Optimization** - LBFGS, BFGS, FIRE
3. **Molecular Dynamics** - NVE, NVT, NPT
4. **Phonon Calculation** - DOS, thermodynamic properties
5. **Mechanical Properties** - Bulk modulus, EOS
6. **Adsorption Energy** - CO₂, H₂O, CH₄, etc.
7. **Coordination Analysis** - Metal coordination environment
8. **High-Throughput Screening** - Batch processing

## 📁 Project Structure

```
mattersim-inference/
├── src/mattersim_inference/
│   ├── __init__.py
│   ├── core.py           # Main class MatterSimInference
│   ├── cli.py            # Command-line interface
│   ├── utils/            # Utility modules
│   │   ├── device.py     # Device management
│   │   └── io.py         # I/O operations
│   └── tasks/            # Task modules
│       ├── static.py     # Single-point calculations
│       ├── dynamics.py   # Molecular dynamics
│       ├── phonon.py     # Phonon calculations
│       ├── mechanics.py  # Mechanical properties
│       └── adsorption.py # Adsorption energy
├── examples/             # Example scripts
├── tests/                # Tests
├── pyproject.toml
└── README.md
```

## 📚 Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [INSTALL_GUIDE.md](INSTALL_GUIDE.md) - Detailed installation instructions
- [MatterSim_inference_tasks.md](../MatterSim_inference_tasks.md) - Inference task guide
- [MatterSim_inference_API_reference.md](../MatterSim_inference_API_reference.md) - API reference

## 📖 Citation

```bibtex
@article{yang2024mattersim,
  title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
  author={Yang, Han and others},
  journal={arXiv preprint arXiv:2405.04967},
  year={2024}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📧 Contact

For questions, please contact us via GitHub Issues or email: shadow.li981@gmail.com
