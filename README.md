# MLFF-inference

**Machine Learning Force Field Inference Toolkit**

A comprehensive collection of inference packages for state-of-the-art Machine Learning Force Fields (MLFFs), optimized for materials science and molecular simulations.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 Overview

This project provides unified, easy-to-use inference interfaces for 7 leading machine learning force field models featured in the MOFSimBench benchmark. Each model package offers:

- 🚀 **Unified API**: Consistent interface across all models
- 🔧 **Rich Functionality**: Single-point calculations, structure optimization, molecular dynamics, phonon calculations, mechanical properties
- 💻 **CPU/GPU Support**: Flexible deployment options
- 📦 **Standalone Packages**: Independent installation and usage
- 🌐 **Comprehensive Documentation**: Both English and Chinese documentation

## 🎯 Supported Models

| Rank | Model | Package | Key Features | Primary Use Cases |
|------|-------|---------|--------------|-------------------|
| 2 | **MACE** | `MACE_inference` | Equivariant message passing, high accuracy | General materials, organic molecules |
| 2 | **Orb** | `Orb_inference` | Fast inference, pre-trained on diverse datasets | Multi-material predictions |
| 1 | **eSCN** | `eSEN_inference` | Equivariant spherical channels, OCP dataset | Catalysis, surface reactions |
| 3 | **MatterSim** | `MatterSim_inference` | M3GNet architecture, uncertainty estimation | MOF adsorption, general materials |
| 4 | **SevenNet** | `SevenNet_inference` | 7-layer equivariant GNN, force accuracy | Molecular simulations, dynamics |
| 5 | **EquiformerV2** | `EquiformerV2_inference` | E(3) equivariant transformer | Large-scale systems, OCP |
| 6 | **GRACE** | `GRACE_inference` | Graph basis functions, DGL backend | MOF gas adsorption, fast computation |

*Rankings based on MOFSimBench performance*

## 🚀 Quick Start

### Installation

Each model package can be installed independently:

```bash
# Example: Install MACE with CPU support
cd MACE_inference/mace-inference
pip install -e ".[cpu]"

# Or with GPU support
pip install -e ".[gpu]"
```

### Basic Usage

All models share a unified API:

```python
from mace_inference import MACEInference
from ase.io import read

# Initialize model
model = MACEInference(model_path="path/to/model.pth", device="cuda")

# Load structure
atoms = read("structure.cif")

# Single-point energy and force calculation
result = model.calculate(atoms)
print(f"Energy: {result['energy']} eV")
print(f"Forces shape: {result['forces'].shape}")

# Structure optimization
optimized = model.optimize(atoms, fmax=0.01)
optimized.write("optimized.cif")

# Molecular dynamics
trajectory = model.run_md(
    atoms,
    temperature=300,
    steps=1000,
    timestep=1.0
)
```

### Command Line Interface

Each package provides a comprehensive CLI:

```bash
# Single-point calculation
mace-inference single-point structure.cif --model model.pth

# Structure optimization
mace-inference optimize structure.cif --fmax 0.01 --output opt.cif

# Molecular dynamics
mace-inference md structure.cif --temp 300 --steps 10000

# Phonon calculation
mace-inference phonon structure.cif --supercell 2 2 2

# Bulk modulus calculation
mace-inference bulk-modulus structure.cif

# Model information
mace-inference info --model model.pth
```

## 📦 Project Structure

```
MLFF-inference/
├── README.md                      # This file (English)
├── README-cn.md                   # Chinese version
├── docs/                          # Documentation
│   └── MOFSimBench_论文分析_*.md
├── MACE_inference/                # MACE model package
│   ├── requirements-cpu.txt
│   ├── requirements-gpu.txt
│   ├── INSTALL.md
│   └── mace-inference/
│       ├── src/mace_inference/
│       ├── examples/
│       ├── tests/
│       └── docs/
├── Orb_inference/                 # Orb model package
├── eSEN_inference/                # eSCN model package
├── MatterSim_inference/           # MatterSim model package
├── SevenNet_inference/            # SevenNet model package
├── EquiformerV2_inference/        # EquiformerV2 model package
└── GRACE_inference/               # GRACE model package
```

## 🔧 Available Tasks

All model packages support the following computational tasks:

### 1. Single-Point Calculations
Calculate energy, forces, and stress for a given structure.

### 2. Structure Optimization
Optimize atomic positions and/or lattice parameters to minimize energy.

### 3. Molecular Dynamics (MD)
- NVE ensemble
- NVT ensemble (Langevin thermostat)
- NPT ensemble (Berendsen barostat)

### 4. Phonon Calculations
Compute phonon dispersion, density of states, and thermodynamic properties using the finite displacement method.

### 5. Mechanical Properties
Calculate elastic constants and bulk modulus through strain-stress relationships.

### 6. Adsorption Energy (Selected Models)
Compute gas adsorption energies on MOF structures (MatterSim, GRACE).

## 📚 Documentation

Each model package includes comprehensive documentation:

- **README.md**: Overview and quick start
- **QUICKSTART.md**: Step-by-step tutorial
- **INSTALL_GUIDE.md**: Detailed installation instructions
- **INSTALL.md**: 安装指南（中文）
- **{Model}_API_reference.md**: API 参考文档（中文）
- **{Model}_tasks.md**: 任务说明文档（中文）

## 💻 System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8 GB RAM
- 10 GB disk space

### Recommended for GPU
- CUDA 11.8 or 12.1
- 16 GB RAM
- NVIDIA GPU with 8+ GB VRAM

### Supported Platforms
- Linux (Ubuntu 20.04+, CentOS 7+)
- macOS (10.15+)
- Windows 10/11

## 🛠️ Development

### Running Tests

```bash
cd {model}-inference
pytest tests/
```

### Code Structure

Each model package follows a consistent structure:

```python
# Core inference class
class {Model}Inference:
    def __init__(self, model_path, device="cpu")
    def calculate(self, atoms)
    def optimize(self, atoms, fmax=0.05)
    def run_md(self, atoms, temperature, steps)
    def calculate_phonon(self, atoms, supercell)
    def calculate_bulk_modulus(self, atoms)
    
# Utility modules
utils/
├── device.py      # Device management
└── io.py          # File I/O operations

# Task modules
tasks/
├── static.py      # Single-point and optimization
├── dynamics.py    # Molecular dynamics
├── phonon.py      # Phonon calculations
└── mechanics.py   # Mechanical properties
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file in each package for details.

## 🙏 Acknowledgments

- **MOFSimBench**: Benchmark framework for evaluating MLFF models on MOF systems
- **ASE**: Atomic Simulation Environment for structure manipulation
- Individual model developers and their respective teams:
  - MACE team (University of Cambridge, École Polytechnique Fédérale de Lausanne)
  - Orb team (Orbital Materials)
  - eSCN/OCP team (Meta AI Research)
  - MatterSim team (Microsoft Research)
  - SevenNet team
  - EquiformerV2/OCP team (Meta AI Research)
  - GRACE team

## 📞 Contact

For questions, issues, or suggestions:

- Open an issue on GitHub
- Email: shadow.li981@gmail.com
- Check individual package documentation
- Refer to the original model repositories

## 🔗 References

1. MOFSimBench: Benchmark for Machine Learning Force Fields on MOF Systems
2. MACE: Higher Order Equivariant Message Passing Neural Networks
3. Orb: Pre-trained Models for Materials Science
4. eSCN: Equivariant Spherical Channel Networks
5. MatterSim: Deep Learning Potentials for Materials
6. SevenNet: Multi-layer Equivariant Graph Neural Networks
7. EquiformerV2: E(3) Equivariant Transformer
8. GRACE: Graph Basis Functions for Materials

## 📊 Citation

If you use this toolkit in your research, please cite the relevant model papers and:

```bibtex
@software{mlff_inference,
  title={MLFF-inference: Machine Learning Force Field Inference Toolkit},
  author={Shibo Li},
  year={2026},
  url={https://github.com/lichman0405/mlff-inference}
}
```

---

**Note**: This is an inference-only toolkit. For model training, please refer to the original model repositories.
