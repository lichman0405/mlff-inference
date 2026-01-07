# MACE Inference - Quick Start Guide

## Installation

```bash
# Basic installation (CPU)
pip install -e .

# With D3 dispersion correction
pip install -e ".[d3]"

# With GPU support
pip install -e ".[gpu]"

# Full installation (all features)
pip install -e ".[all]"
```

## Basic Usage

### Python API

```python
from mace_inference import MACEInference

# Initialize calculator
calc = MACEInference(model="medium", device="auto")

# Single-point energy
result = calc.single_point("structure.cif")
print(f"Energy: {result['energy']:.6f} eV")

# Structure optimization
optimized = calc.optimize("structure.cif", fmax=0.05, output="optimized.cif")

# Molecular dynamics
final = calc.run_md("structure.cif", ensemble="nvt", temperature_K=300, steps=1000)

# Phonon calculation
phonon = calc.phonon("structure.cif", supercell_matrix=[2, 2, 2])

# Bulk modulus
bm = calc.bulk_modulus("structure.cif")
print(f"Bulk Modulus: {bm['B_GPa']:.2f} GPa")

# Adsorption energy
E_ads = calc.adsorption_energy(
    mof_structure="mof.cif",
    gas_molecule="CO2",
    site_position=[10.0, 10.0, 10.0]
)
print(f"E_ads = {E_ads['E_ads']:.3f} eV")
```

### Command Line Interface

```bash
# Single-point energy
mace-infer energy structure.cif --model medium

# Structure optimization
mace-infer optimize structure.cif --fmax 0.05 --output optimized.cif

# Molecular dynamics
mace-infer md structure.cif --ensemble nvt --temp 300 --steps 10000 --trajectory md.traj

# Phonon calculation
mace-infer phonon structure.cif --supercell 2 2 2 --temp-range 0 1000 10

# Bulk modulus
mace-infer bulk-modulus structure.cif

# Adsorption energy
mace-infer adsorption mof.cif --gas CO2 --site 10.0 10.0 10.0 --d3

# Show system info
mace-infer info --verbose
```

## Examples

See the `examples/` directory:

- `01_basic_usage.py` - Single-point and optimization
- `02_molecular_dynamics.py` - NVT/NPT simulations
- `03_phonon_calculation.py` - Phonon and thermal properties
- `04_adsorption_study.py` - Gas adsorption in MOFs
- `05_high_throughput.py` - Batch processing

Run an example:

```bash
cd examples
python 01_basic_usage.py
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mace_inference --cov-report=html

# Run specific test file
pytest tests/test_utils.py -v
```

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/
```

## Available Models

- **small**: MACE-MP-0 small (fast, less accurate)
- **medium**: MACE-MP-0 medium (balanced, **recommended**)
- **large**: MACE-MP-0 large (slow, most accurate)
- Or path to custom model file

## Device Selection

- **auto**: Automatically detect (use GPU if available)
- **cpu**: Force CPU (no CUDA required)
- **cuda**: Force GPU (requires CUDA)

## Performance Tips

1. **GPU**: Use `device="cuda"` for 10-100x speedup on large systems
2. **Batch**: Process multiple structures with same calculator instance
3. **Supercells**: For MD, use at least 3x3x3 supercells
4. **D3 Correction**: Enable for systems with van der Waals interactions

## Citation

If you use this library, please cite:

```bibtex
@software{mace_inference,
  title = {MACE Inference: High-level Python library for MACE force field inference},
  author = {MACE Inference Contributors},
  year = {2026},
  url = {https://github.com/yourusername/mace-inference}
}
```

And the MACE paper:

```bibtex
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
