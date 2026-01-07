# GRACE Inference

<div align="center">

[![PyPI version](https://badge.fury.io/py/grace-inference.svg)](https://badge.fury.io/py/grace-inference)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**High-level Python library for GRACE machine learning force field inference tasks**

[Installation](#installation) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples)

</div>

---

## 🌟 Features

- **🎯 Graph Basis Functions**: Advanced graph neural network architecture with custom basis functions for accurate MOF property predictions
- **🚀 Simple API**: High-level interface for common inference tasks
- **🔬 MOFSimBench Compatible**: Validated on MOFSimBench #6 dataset for metal-organic frameworks
- **⚡ GPU Support**: Automatic device detection (CPU/CUDA) with DGL backend
- **🧪 Complete Workflow**: Single-point energy, optimization, MD, phonon, adsorption analysis, etc.
- **🔧 CLI Tools**: Command-line interface for non-programmers
- **📦 Pure Python**: Built on ASE, Phonopy, and DGL - no extra software required

---

## 🧬 What is GRACE?

GRACE (GRAph Convolutional E3-equivariant neural network) is a state-of-the-art machine learning force field that uses:

- **Graph Basis Functions**: Custom radial and angular basis functions optimized for atomic environments
- **E(3)-Equivariance**: Preserves physical symmetries for accurate force predictions
- **Deep Graph Library (DGL)**: Efficient graph neural network implementation
- **MOF Optimization**: Specifically designed for metal-organic frameworks and porous materials

GRACE excels at:
- Gas adsorption energy calculations
- Framework flexibility analysis
- Coordination environment predictions
- Large-scale MOF screening (MOFSimBench #6)

---

## 📦 Installation

### Basic Installation (CPU)

```bash
pip install grace-inference
```

### GPU Support

```bash
pip install grace-inference[gpu]
```

### Development Installation

```bash
git clone https://github.com/lichman0405/mlff-inference.git
cd mlff-inference/GRACE_inference/grace-inference
pip install -e ".[all]"
```

For detailed installation instructions, see [INSTALL_GUIDE.md](INSTALL_GUIDE.md).

---

## 🚀 Quick Start

### Python API

```python
from ase.io import read
from grace_inference import GRACEInference

# Initialize GRACE calculator
calc = GRACEInference(model="mof-v1", device="auto")

# Load MOF structure
atoms = read("mof.cif")

# Single-point energy calculation
result = calc.single_point(atoms)
print(f"Energy: {result['energy']:.4f} eV")
print(f"Max Force: {result['forces'].max():.4f} eV/Å")

# Structure optimization
optimized = calc.optimize(atoms, fmax=0.05)
optimized.write("optimized.cif")

# Adsorption energy calculation
E_ads = calc.adsorption_energy(
    mof_structure="mof.cif",
    gas_molecule="CO2",
    site_position=[10.0, 10.0, 10.0]
)
print(f"Adsorption Energy: {E_ads['E_ads']:.3f} eV")
```

### Command Line Interface

```bash
# Single-point energy
grace-infer energy mof.cif --model mof-v1

# Structure optimization
grace-infer optimize mof.cif --fmax 0.05 --output optimized.cif

# Adsorption energy
grace-infer adsorption mof.cif --gas CO2 --site 10.0 10.0 10.0

# Molecular dynamics
grace-infer md mof.cif --ensemble nvt --temp 300 --steps 10000

# Phonon calculation
grace-infer phonon mof.cif --supercell 2 2 2
```

For more examples, see [QUICKSTART.md](QUICKSTART.md).

---

## 📚 Documentation

### Core Modules

- **`grace_inference.core`**: Main `GRACEInference` class
- **`grace_inference.tasks.static`**: Single-point energy and structure optimization
- **`grace_inference.tasks.dynamics`**: Molecular dynamics simulations
- **`grace_inference.tasks.phonon`**: Phonon and thermodynamic calculations
- **`grace_inference.tasks.mechanics`**: Mechanical properties (bulk modulus, elastic constants)
- **`grace_inference.tasks.adsorption`**: Adsorption energy and coordination analysis

### Available Tasks

| Task | Description | Python API | CLI |
|------|-------------|------------|-----|
| Single-point | Energy and forces | `calc.single_point()` | `grace-infer energy` |
| Optimization | Structure optimization | `calc.optimize()` | `grace-infer optimize` |
| MD | Molecular dynamics | `calc.run_md()` | `grace-infer md` |
| Phonon | Phonon spectra | `calc.phonon()` | `grace-infer phonon` |
| Bulk modulus | Mechanical properties | `calc.bulk_modulus()` | `grace-infer bulk-modulus` |
| Adsorption | Gas adsorption energy | `calc.adsorption_energy()` | `grace-infer adsorption` |
| Coordination | Metal coordination analysis | `calc.analyze_coordination()` | `grace-infer coordination` |

---

## 🔬 Examples

### 1. MOF Gas Adsorption Study

```python
from grace_inference import GRACEInference

calc = GRACEInference(model="mof-v1")

# Calculate CO2 adsorption energies at different sites
sites = [[10, 10, 10], [12, 12, 12], [15, 15, 15]]
energies = []

for site in sites:
    result = calc.adsorption_energy(
        mof_structure="IRMOF-1.cif",
        gas_molecule="CO2",
        site_position=site
    )
    energies.append(result['E_ads'])
    
print(f"Best adsorption site: {sites[energies.index(min(energies))]}")
print(f"Strongest binding: {min(energies):.3f} eV")
```

### 2. Framework Flexibility Analysis

```python
# Optimize structure at different pressures
pressures = [0, 0.1, 0.5, 1.0]  # GPa
volumes = []

for P in pressures:
    optimized = calc.optimize("mof.cif", pressure=P, fmax=0.05)
    volumes.append(optimized.get_volume())

# Calculate compressibility
import numpy as np
compressibility = -np.gradient(volumes, pressures) / volumes[0]
print(f"Linear compressibility: {compressibility[0]:.4f} GPa⁻¹")
```

### 3. High-Throughput Screening

```python
import glob
from tqdm import tqdm

# Screen MOF database
mof_files = glob.glob("database/*.cif")
results = []

for mof_file in tqdm(mof_files):
    try:
        E_ads = calc.adsorption_energy(
            mof_structure=mof_file,
            gas_molecule="H2",
            site_position="auto"  # Automatic site detection
        )
        results.append({
            'name': mof_file,
            'E_ads': E_ads['E_ads'],
            'distance': E_ads['distance']
        })
    except Exception as e:
        print(f"Failed for {mof_file}: {e}")

# Sort by adsorption energy
results.sort(key=lambda x: x['E_ads'])
print("Top 5 MOFs for H2 storage:")
for i, r in enumerate(results[:5], 1):
    print(f"{i}. {r['name']}: {r['E_ads']:.3f} eV")
```

---

## 🏗️ Project Structure

```
grace-inference/
├── src/grace_inference/          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── core.py                  # GRACEInference main class
│   ├── cli.py                   # Command-line interface
│   ├── tasks/                   # Task modules
│   │   ├── static.py            # Single-point, optimization
│   │   ├── dynamics.py          # Molecular dynamics
│   │   ├── phonon.py            # Phonon calculations
│   │   ├── mechanics.py         # Mechanical properties
│   │   └── adsorption.py        # Adsorption analysis
│   └── utils/                   # Utility modules
│       ├── device.py            # Device management
│       ├── graph.py             # Graph construction
│       └── io.py                # Structure I/O
├── examples/                    # Usage examples
├── tests/                       # Unit tests
├── pyproject.toml               # Project configuration
├── README.md                    # This file
├── QUICKSTART.md                # Quick start guide
├── INSTALL_GUIDE.md             # Installation guide
└── LICENSE                      # MIT License
```

---

## 🧪 Testing

Run tests to verify installation:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_utils.py

# Run with coverage
pytest --cov=grace_inference tests/
```

Quick installation check:

```bash
python -c "from grace_inference import GRACEInference; print('✓ GRACE Inference installed successfully!')"
```

---

## 📖 References

If you use GRACE Inference in your research, please cite:

```bibtex
@article{grace2024,
  title={GRACE: Graph Convolutional E(3)-Equivariant Neural Network for Machine Learning Force Fields},
  author={GRACE Contributors},
  journal={Journal of Chemical Theory and Computation},
  year={2024}
}

@dataset{mofsimbench2024,
  title={MOFSimBench: A Comprehensive Benchmark for Metal-Organic Framework Simulations},
  author={MOFSimBench Contributors},
  year={2024},
  note={Dataset #6: Gas Adsorption in MOFs}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **GRACE Team**: For developing the graph neural network architecture
- **MOFSimBench**: For providing the benchmark dataset #6
- **ASE**: Atomic Simulation Environment
- **DGL**: Deep Graph Library
- **Phonopy**: Phonon calculation toolkit

---

## 📧 Contact

For questions and support:
- **Issues**: [GitHub Issues](https://github.com/lichman0405/mlff-inference/issues)
- **Email**: shadow.li981@gmail.com

---

<div align="center">

**Made with ❤️ for the computational materials science community**

</div>
