# Orb Inference

**High-level Python library for materials science calculations using Orb's GNS-based machine learning force fields.**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Orb Inference provides a unified, high-level interface for performing materials science calculations with **Orb models** developed by Orbital Materials. Orb uses a Graph Network Simulator (GNS) architecture that learns equivariance through data augmentation rather than predefined symmetries, enabling flexible and accurate predictions for diverse materials including metal-organic frameworks (MOFs).

### Key Features

- **Unified API**: Single `OrbInference` class for all tasks
- **8 Task Categories**: Energy, optimization, MD, phonons, mechanics, adsorption, and more
- **Multiple Models**: Support for orb-v3 (conservative forces) and orb-v2 (D3 dispersion)
- **Command-Line Interface**: `orb-infer` CLI for quick calculations
- **Production-Ready**: Optimized for MOF simulations and high-throughput screening

### Orb Models

| Model | Training Data | Features | Use Case |
|-------|---------------|----------|----------|
| **orb-v3-omat** | OMAT24 | Conservative forces, no neighbor limit | **Recommended** for general use |
| **orb-v3-mpa** | MPtraj + Alexandria | Conservative forces | Materials Project focus |
| **orb-d3-v2** | MPtraj | D3 dispersion (built-in) | Dispersion-critical systems |
| **orb-mptraj-only-v2** | MPtraj only | Older version | Legacy support |

**v3 vs v2**: v3 uses conservative forces (energy gradients), has no neighbor limit, and is recommended for production use. v2 includes built-in D3 dispersion but limits to 30 neighbors.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/MLFF-inference/orb-inference
cd orb-inference

# Install (CPU)
pip install -e .

# Install (GPU with CUDA 12.1)
pip install -r requirements-gpu.txt
pip install -e .
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Basic Usage

```python
from orb_inference import OrbInference

# Initialize model (auto-downloads on first run)
orb = OrbInference(model_name='orb-v3-omat', device='cuda')

# Single-point energy
result = orb.single_point('structure.cif')
print(f"Energy: {result['energy']:.4f} eV")

# Structure optimization
opt_result = orb.optimize('structure.cif', fmax=0.01)
optimized = opt_result['atoms']

# Molecular dynamics (300 K, 5000 steps)
final = orb.run_md(optimized, temperature=300, steps=5000, ensemble='nvt')
```

### Command-Line Interface

```bash
# Single-point energy
orb-infer energy structure.cif --model orb-v3-omat

# Optimize structure
orb-infer optimize structure.cif -o optimized.cif --fmax 0.01 --relax-cell

# Molecular dynamics (NPT, 300 K, 1 atm)
orb-infer md structure.cif -T 300 -P 0 -n 5000 --ensemble npt -t md.traj

# Phonon calculation
orb-infer phonon structure.cif --supercell 2,2,2 --mesh 20,20,20

# Bulk modulus
orb-infer bulk-modulus structure.cif --strain 0.05 --points 7

# Model information
orb-infer info --model orb-v3-omat
```

## Documentation

- **[Task Guide](Orb_inference_tasks.md)**: 8 task categories with examples and benchmarks
- **[API Reference](Orb_inference_API_reference.md)**: Complete API documentation
- **[Installation Guide](INSTALL.md)**: Detailed setup for CPU/GPU environments
- **[Quick Start Guide](QUICKSTART.md)**: 5-minute tutorial

## Examples

See [`examples/`](examples/) directory:

1. [`01_basic_usage.py`](examples/01_basic_usage.py) - Model initialization and optimization
2. [`02_molecular_dynamics.py`](examples/02_molecular_dynamics.py) - NVT/NPT simulations
3. [`03_phonon_calculation.py`](examples/03_phonon_calculation.py) - Phonons and thermal properties
4. [`04_mechanical_properties.py`](examples/04_mechanical_properties.py) - Bulk modulus and EOS
5. [`05_high_throughput.py`](examples/05_high_throughput.py) - Batch processing

## Performance

**MOFSimBench Rankings** (out of 7 model families):

| Task | Orb Ranking | MAE |
|------|-------------|-----|
| **Overall** | #2 | - |
| Energy | #2 | 0.039 eV/atom |
| Forces | #2 | 0.089 eV/Å |
| Stress | #3 | 0.31 GPa |
| **Heat Capacity** | **#1** | 0.051 J/(K·g) |
| Bulk Modulus | #3 | 5.16 GPa |

Orb excels at thermodynamic properties (especially heat capacity) due to its flexible GNS architecture.

## Architecture

```
orb-inference/
├── src/orb_inference/
│   ├── core.py              # OrbInference main class
│   ├── cli.py               # Command-line interface
│   ├── utils/               # Device, I/O utilities
│   │   ├── device.py
│   │   └── io.py
│   └── tasks/               # Task-specific modules
│       ├── static.py        # Energy, optimization
│       ├── dynamics.py      # MD (NVT/NPT)
│       ├── phonon.py        # Phonon, thermal properties
│       ├── mechanics.py     # Bulk modulus, EOS
│       └── adsorption.py    # Adsorption, coordination
├── examples/                # 5 usage examples
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## Requirements

- **Python**: 3.8 - 3.11 (3.10 recommended)
- **PyTorch**: 2.3.1 (avoid 2.4.1 due to compatibility issues)
- **ASE**: ≥ 3.23.0 (for `FrechetCellFilter`)
- **Phonopy**: ≥ 2.20.0
- **orb-models**: Latest (auto-installs from PyPI)

## Citation

If you use Orb Inference in your research, please cite:

```bibtex
@software{orb_inference,
  title = {Orb Inference: High-Level Interface for Orb Machine Learning Force Fields},
  author = {MLFF-inference Project},
  year = {2024},
  url = {https://github.com/MLFF-inference/orb-inference}
}
```

And the original Orb model:

```bibtex
@article{orb2024,
  title = {Orb: A Fast, Scalable Neural Network for Materials},
  author = {Orbital Materials},
  year = {2024},
  url = {https://orbitalmaterials.com}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

- **Issues**: [GitHub Issues](https://github.com/MLFF-inference/orb-inference/issues)
- **Documentation**: [Full docs](Orb_inference_tasks.md)
- **Orb Models**: [Orbital Materials](https://orbitalmaterials.com)

## Acknowledgments

- **Orbital Materials** for developing Orb models
- **ASE** (Atomic Simulation Environment) for structure manipulation
- **Phonopy** for phonon calculations
- **MOFSimBench** for benchmark datasets
