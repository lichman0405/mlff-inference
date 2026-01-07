# eSEN Inference - Python Inference Library for eSEN Models

[![PyPI Version](https://img.shields.io/pypi/v/esen-inference)](https://pypi.org/project/esen-inference/)
[![Python Versions](https://img.shields.io/pypi/pyversions/esen-inference)](https://pypi.org/project/esen-inference/)
[![License](https://img.shields.io/github/license/yourusername/esen-inference)](LICENSE)
[![MOFSimBench Rank](https://img.shields.io/badge/MOFSimBench-Rank%20%231-gold)](https://github.com/SanggyuChong/MOFSimBench)

**eSEN Inference** is a production-ready Python library for running inference with **eSEN (Smooth & Expressive Equivariant Networks)** machine learning force fields - the **#1 ranked model** in the MOFSimBench benchmark.

## 🏆 eSEN Performance Highlights

**MOFSimBench Overall Ranking: #1** 🥇

| Task | Ranking | Performance |
|------|---------|-------------|
| **Energy Prediction** | **#1** 🥇 | MAE 0.041 eV/atom |
| **Bulk Modulus** | **#1** 🥇 | MAE 2.64 GPa |
| **Structure Optimization** | **#1** 🥇 | 89% success rate |
| **MD Stability** | **#1** 🥇 | Excellent |
| Force Prediction | #2 🥈 | MAE 0.084 eV/Å |
| Adsorption Energy | #2 🥈 | Excellent |
| Heat Capacity | #3 🥉 | MAE 0.024 J/(K·g) |

**Key Strengths**:
- ✅ **Narrowest error distribution** across all tasks
- ✅ **Best energy prediction** accuracy
- ✅ **Best mechanical properties** (bulk modulus)
- ✅ **Best MD stability** for long simulations
- ✅ **Highest optimization success rate**

## 📦 Features

- **8 Inference Tasks**:
  1. Single-point energy/force/stress calculations
  2. Structure optimization (coordinates & cell)
  3. Molecular dynamics (NVE/NVT/NPT)
  4. Phonon & thermodynamic properties
  5. Mechanical properties (bulk modulus, elastic constants)
  6. Adsorption energy calculations
  7. Coordination analysis
  8. High-throughput screening

- **2 Pre-trained Models**:
  - `esen-30m-oam`: **OMat24 + MPtraj + sAlex** (Recommended for all MOFs)
  - `esen-30m-mp`: MPtraj only (Materials Project specialized)

- **Production Features**:
  - GPU/CPU support with automatic device management
  - Float32/Float64 precision options
  - ASE integration for seamless workflows
  - Phonopy integration for phonon calculations
  - CLI tools for batch processing
  - Comprehensive API documentation

## 🚀 Quick Start

### Installation

```bash
# Create conda environment
conda create -n esen python=3.10 -y
conda activate esen

# Install PyTorch with CUDA 11.8
conda install pytorch==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install eSEN Inference
pip install esen-inference
```

### Basic Usage

```python
from esen_inference import ESENInference
from ase.io import read

# Initialize eSEN model (auto-downloads checkpoint on first use)
esen = ESENInference(
    model_name='esen-30m-oam',  # Recommended: OAM version
    device='cuda'                # 'cuda' or 'cpu'
)

# Load MOF structure
atoms = read('MOF-5.cif')

# Single-point calculation
result = esen.single_point(atoms)
print(f"Energy: {result['energy']:.6f} eV")
print(f"Max force: {result['max_force']:.6f} eV/Å")

# Structure optimization
opt_result = esen.optimize(atoms, fmax=0.01, relax_cell=True)
print(f"Converged in {opt_result['steps']} steps")

# Molecular dynamics (NVT, 300 K, 50 ps)
final_atoms = esen.run_md(
    atoms,
    temperature=300.0,
    steps=50000,
    timestep=1.0,
    ensemble='nvt'
)

# Bulk modulus
bulk_result = esen.bulk_modulus(atoms, strain_range=0.05)
print(f"Bulk modulus: {bulk_result['bulk_modulus']:.2f} GPa")
```

### Command-Line Interface

```bash
# Single-point calculation
esen-infer single-point MOF-5.cif --output result.json

# Structure optimization
esen-infer optimize MOF-5.cif --fmax 0.01 --relax-cell --output MOF-5_opt.cif

# Batch processing
esen-infer batch-optimize mof_database/*.cif --output-dir optimized/

# Molecular dynamics
esen-infer md MOF-5_opt.cif --temperature 300 --steps 50000 --ensemble nvt

# Phonon calculation
esen-infer phonon MOF-5_primitive.cif --supercell 2 2 2 --mesh 20 20 20

# Bulk modulus
esen-infer bulk-modulus MOF-5_opt.cif --strain-range 0.05 --n-points 7
```

## 📚 Documentation

- **[eSEN_inference_tasks.md](eSEN_inference_tasks.md)**: Detailed guide for all 8 inference tasks
- **[eSEN_inference_API_reference.md](eSEN_inference_API_reference.md)**: Complete API documentation
- **[eSEN_inference_INSTALL.md](eSEN_inference_INSTALL.md)**: Installation guide and troubleshooting

## 🔬 eSEN Model Details

**eSEN (Smooth & Expressive Equivariant Networks)** is an E(3)-equivariant GNN developed by Meta FAIR (Fu et al. 2025).

### Key Innovations

1. **Smoothness**: Ensures smooth potential energy surfaces for stable MD simulations
2. **Expressiveness**: Balances smoothness with high representational power
3. **E(3) Equivariance**: Rigorously preserves rotational, translational, and reflection symmetry
4. **Conservative Forces**: Forces computed as energy gradients for physical consistency

### Model Architecture

- **Type**: E(3)-Equivariant Graph Neural Network
- **Parameters**: 30M (medium-sized)
- **Training Data**:
  - **eSEN-30M-OAM**: OMat24 + MPtraj + sAlex (867K+ structures)
  - **eSEN-30M-MP**: MPtraj only (Materials Project)
- **Supported Elements**: All 118 elements
- **Output Properties**: Energy, forces, stress tensor

### References

- **Paper**: [arXiv:2502.12147](https://arxiv.org/abs/2502.12147) (Fu et al. 2025)
- **Code**: [FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem)
- **Benchmark**: [MOFSimBench](https://github.com/SanggyuChong/MOFSimBench)

## 💡 Examples

### Example 1: High-Throughput Screening

```python
from esen_inference import ESENInference
from ase.io import read, write
from pathlib import Path
import numpy as np

# Initialize once, use for all MOFs
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

results = {}
for mof_file in Path('mof_database').glob('*.cif'):
    atoms = read(mof_file)
    
    # Optimize structure
    opt_result = esen.optimize(atoms, fmax=0.05, relax_cell=True)
    if not opt_result['converged']:
        continue
    
    # Calculate bulk modulus
    bulk_result = esen.bulk_modulus(opt_result['atoms'], optimize_first=False)
    
    results[mof_file.stem] = {
        'energy': opt_result['final_energy'],
        'bulk_modulus': bulk_result['bulk_modulus']
    }

# Find hardest MOF
hardest = max(results.items(), key=lambda x: x[1]['bulk_modulus'])
print(f"Hardest MOF: {hardest[0]} (B = {hardest[1]['bulk_modulus']:.2f} GPa)")
```

### Example 2: Adsorption Energy Calculation

```python
from esen_inference import ESENInference
from ase.io import read

esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# Load structures
host = read('HKUST-1.cif')
guest = read('CO2.xyz')
complex_atoms = read('HKUST-1_CO2.cif')

# Calculate adsorption energy
result = esen.adsorption_energy(
    host=host,
    guest=guest,
    complex_atoms=complex_atoms,
    optimize_complex=True
)

E_ads_kJ_mol = result['E_ads'] * 96.485
print(f"CO₂ adsorption energy: {E_ads_kJ_mol:.2f} kJ/mol")
```

### Example 3: Phonon Calculation & Heat Capacity

```python
from esen_inference import ESENInference
from esen_inference.tasks.phonon import plot_phonon_dos, plot_thermal_properties
from ase.io import read

esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# Load optimized primitive cell
primitive = read('MOF-5_primitive_opt.cif')

# Phonon calculation
result = esen.phonon(
    primitive,
    supercell_matrix=[2, 2, 2],
    mesh=[20, 20, 20],
    t_min=0,
    t_max=1000
)

# Check for imaginary modes
if not result['has_imaginary']:
    print("✓ Structure is dynamically stable")
    
    # Plot phonon DOS
    plot_phonon_dos(result['frequency_points'], result['total_dos'], 'phonon_dos.png')
    
    # Get heat capacity at 300 K
    thermal = result['thermal']
    idx_300K = (thermal['temperatures'] >= 300).argmax()
    print(f"Cv at 300 K: {thermal['heat_capacity'][idx_300K]:.2f} J/(K·mol)")
```

## 🛠️ Development

### From Source

```bash
git clone https://github.com/yourusername/esen-inference.git
cd esen-inference
conda env create -f environment.yml
conda activate esen
pip install -e .
```

### Running Tests

```bash
pytest tests/ -v
```

## 📊 Performance Benchmarks

### MOFSimBench Results (eSEN-30M-OAM)

| Task Category | Metric | Value | Rank |
|---------------|--------|-------|------|
| Energy | MAE (eV/atom) | **0.041** | **#1** |
| Force | MAE (eV/Å) | 0.084 | #2 |
| Stress | MAE (GPa) | 0.31 | #3 |
| Bulk Modulus | MAE (GPa) | **2.64** | **#1** |
| Heat Capacity | MAE (J/(K·g)) | 0.024 | #3 |
| Optimization | Success Rate | **89%** | **#1** |
| MD Stability | 20 ps | **Excellent** | **#1** |

### Recommended Use Cases

| Scenario | Recommended Model | Reason |
|----------|-------------------|--------|
| **General MOF modeling** | **eSEN-OAM** | Overall best performance |
| **Mechanical properties** | **eSEN-OAM** | Best bulk modulus (#1) |
| **Long MD simulations** | **eSEN-OAM** | Best stability (#1) |
| **Energy prediction** | **eSEN-OAM** | Best accuracy (#1) |
| **Heat capacity** | orb-v3-omat | #1 in thermal properties |
| **Adsorption** | MatterSim | #1 in adsorption energy |

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **eSEN Development Team**: Meta FAIR - Fu et al. 2025
- **FAIR-Chem Framework**: [FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem)
- **MOFSimBench**: [SanggyuChong/MOFSimBench](https://github.com/SanggyuChong/MOFSimBench)
- **ASE**: [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/)
- **Phonopy**: [Phonopy](https://phonopy.github.io/phonopy/)

## 📧 Contact

- **Issues**: https://github.com/yourusername/esen-inference/issues
- **Discussions**: https://github.com/yourusername/esen-inference/discussions

## 🔗 Related Projects

- [MACE Inference](../MACE_inference/) - MACE model inference (#2 in MOFSimBench)
- [Orb Inference](../Orb_inference/) - Orb model inference (#2 in MOFSimBench)
- [MOFSimBench](https://github.com/SanggyuChong/MOFSimBench) - MOF ML force field benchmark

---

**eSEN Inference v1.0.0** | **MOFSimBench Rank #1** 🥇 | **January 2026**
