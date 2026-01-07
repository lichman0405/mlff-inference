# eSEN-Inference Quick Start Guide

Get started with eSEN-Inference in 5 minutes!

## Installation

### Option 1: pip (Recommended)

```bash
pip install esen-inference
```

### Option 2: From Source

```bash
git clone https://github.com/your-org/esen-inference.git
cd esen-inference
pip install -e .
```

## Basic Usage

### Python API

```python
from esen_inference import ESENInference
from ase.build import bulk

# 1. Initialize model
esen = ESENInference(
    model_name='esen-30m-oam',  # MOFSimBench #1 model
    device='cuda'                # Use 'cpu' if no GPU
)

# 2. Create structure
atoms = bulk('Cu', 'fcc', a=3.6)

# 3. Single-point calculation
result = esen.single_point(atoms)
print(f"Energy: {result['energy']:.6f} eV")
print(f"Max force: {result['max_force']:.6f} eV/Å")

# 4. Optimize structure
opt_result = esen.optimize(atoms, fmax=0.01, relax_cell=True)
print(f"Optimized in {opt_result['steps']} steps")
```

### Command Line Interface

```bash
# Single-point calculation
esen-infer single-point structure.cif --output result.json

# Structure optimization
esen-infer optimize structure.cif --fmax 0.01 --relax-cell

# Molecular dynamics
esen-infer md structure.cif --temperature 300 --steps 50000 --ensemble nvt

# Phonon calculation
esen-infer phonon primitive.cif --supercell 2 2 2 --mesh 20 20 20

# Bulk modulus
esen-infer bulk-modulus structure.cif --strain-range 0.05

# Batch optimization
esen-infer batch-optimize *.cif --output-dir optimized/
```

## Key Features

✅ **MOFSimBench #1 Model**: Best overall performance  
✅ **8 Task Types**: Single-point, optimization, MD, phonon, mechanics, adsorption  
✅ **Easy Integration**: ASE-compatible, FAIR-Chem backend  
✅ **High Performance**: GPU acceleration, batch processing  
✅ **Comprehensive**: Python API + CLI tools

## Available Models

| Model | Training Data | Parameters | Recommended Use |
|-------|---------------|------------|-----------------|
| esen-30m-oam | OMat24 + MPtraj + sAlex | 30M | **Default** (best accuracy) |
| esen-30m-mp | MPtraj only | 30M | Materials Project specific |

## Common Tasks

### Single-Point Energy

```python
result = esen.single_point(atoms)
# Returns: energy, forces, stress, pressure
```

### Structure Optimization

```python
result = esen.optimize(
    atoms,
    fmax=0.01,          # Force convergence threshold
    relax_cell=True,    # Optimize both coords and cell
    optimizer='LBFGS',  # LBFGS/BFGS/FIRE
    max_steps=200
)
optimized_atoms = result['atoms']
```

### Molecular Dynamics

```python
final_atoms = esen.run_md(
    atoms,
    temperature=300,    # K
    steps=50000,        # 50 ps @ 1 fs/step
    timestep=1.0,       # fs
    ensemble='nvt',     # NVE/NVT/NPT
    trajectory='md.traj'
)
```

### Phonon Calculation

```python
result = esen.phonon(
    primitive_cell,
    supercell_matrix=[3, 3, 3],
    mesh=[30, 30, 30],
    t_max=1000
)
heat_capacity = result['thermal']['heat_capacity']
```

### Bulk Modulus

```python
result = esen.bulk_modulus(
    atoms,
    strain_range=0.05,  # ±5% strain
    npoints=11
)
print(f"Bulk modulus: {result['bulk_modulus']:.2f} GPa")
```

## Performance Tips

1. **GPU Usage**: Use `device='cuda'` for 10-100× speedup
2. **Batch Processing**: Process multiple structures in parallel
3. **Precision**: Use `precision='float32'` (default) for speed
4. **Checkpoints**: Models auto-download and cache locally

## Examples

See the `examples/` directory for detailed examples:

- `01_basic_usage.py` - Single-point and optimization
- `02_molecular_dynamics.py` - NVT/NPT MD simulations
- `03_phonon_calculation.py` - Phonon DOS and thermodynamics
- `04_mechanical_properties.py` - Bulk modulus and EOS
- `05_high_throughput.py` - Batch processing workflow

## Troubleshooting

### Model Download Issues

```python
# Manual checkpoint specification
esen = ESENInference(
    model_name='esen-30m-oam',
    checkpoint_path='/path/to/checkpoint.pt'
)
```

### Memory Issues

```python
# Use CPU or reduce batch size
esen = ESENInference(device='cpu')

# Or clear GPU cache
import torch
torch.cuda.empty_cache()
```

### Import Errors

```bash
# Install missing dependencies
pip install torch fairchem-core ase phonopy matplotlib
```

## Next Steps

📖 **Full Documentation**: See `eSEN_inference_API_reference.md`  
📝 **Task Guide**: See `eSEN_inference_tasks.md`  
🔧 **Installation Details**: See `eSEN_inference_INSTALL.md`  
💻 **Examples**: Explore `examples/` directory  
✅ **Tests**: Run `pytest tests/` to verify installation

## Performance Benchmarks (MOFSimBench)

eSEN-30M-OAM is **#1 overall** with:

- Energy MAE: **0.041 eV/atom** (#1)
- Bulk modulus MAE: **2.64 GPa** (#1)
- Optimization success: **89%** (#1)
- Heat capacity MAE: 0.024 J/(K·g) (#3)
- MD stability: Excellent (#1)

## Citation

If you use eSEN in your research, please cite:

```bibtex
@article{fu2025esen,
  title={eSEN: Efficient Equivariant Graph Neural Networks for Atomistic Simulations},
  author={Fu, Xiaoxun and others},
  journal={arXiv preprint arXiv:2502.12147},
  year={2025}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/esen-inference/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/esen-inference/discussions)
- **Email**: support@your-org.com

## License

This project is licensed under the MIT License.

---

**Happy computing with eSEN! 🚀**
