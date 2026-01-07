# Quick Start Guide - Orb Inference

Get started with Orb Inference in 5 minutes!

## Installation

```bash
# Basic installation (CPU)
pip install -e .

# GPU installation (CUDA 12.1)
pip install -r requirements-gpu.txt
pip install -e .
```

**Verify installation:**

```bash
python tests/test_install.py
```

## Your First Calculation

Create `my_first_calc.py`:

```python
from orb_inference import OrbInference
from ase.build import bulk

# 1. Create structure
atoms = bulk('Cu', 'fcc', a=3.6)

# 2. Initialize Orb (downloads model on first run)
orb = OrbInference(model_name='orb-v3-omat', device='cuda')

# 3. Calculate energy
result = orb.single_point(atoms)
print(f"Energy: {result['energy']:.4f} eV")
print(f"Max force: {result['max_force']:.4f} eV/Å")
```

Run it:

```bash
python my_first_calc.py
```

**Output:**
```
Loading Orb model: orb-v3-omat
  Device: cuda
  Precision: float32-high
Model loaded successfully!
  GPU: NVIDIA RTX 4090
  Memory: 0.52 / 24.00 GB

Energy: -3.5482 eV
Max force: 0.0012 eV/Å
```

## Common Tasks

### 1. Structure Optimization

```python
# Optimize structure with cell relaxation
opt_result = orb.optimize(
    'structure.cif',
    fmax=0.01,           # Convergence: max force < 0.01 eV/Å
    relax_cell=True,     # Relax cell vectors
    output='opt.traj'    # Save trajectory
)

print(f"Converged in {opt_result['steps']} steps")
optimized = opt_result['atoms']
```

**CLI equivalent:**
```bash
orb-infer optimize structure.cif -o optimized.cif --fmax 0.01 --relax-cell
```

### 2. Molecular Dynamics

```python
# Run NVT MD at 300 K
final_atoms = orb.run_md(
    atoms,
    temperature=300,     # K
    steps=5000,          # MD steps
    timestep=1.0,        # fs
    ensemble='nvt',
    trajectory='md.traj'
)

# NPT MD (constant pressure)
final_atoms = orb.run_md(
    atoms,
    temperature=300,
    pressure=0.0,        # GPa (ambient)
    steps=5000,
    ensemble='npt',
    trajectory='npt.traj'
)
```

**CLI equivalent:**
```bash
orb-infer md structure.cif -T 300 -n 5000 --ensemble nvt -t md.traj
orb-infer md structure.cif -T 300 -P 0 -n 5000 --ensemble npt -t npt.traj
```

### 3. Phonon & Thermal Properties

```python
# Calculate phonon properties
result = orb.phonon(
    atoms,
    supercell_matrix=[2, 2, 2],  # 2x2x2 supercell
    mesh=[20, 20, 20],           # k-point mesh
    t_min=0, t_max=1000, t_step=10
)

# Access results
temperatures = result['thermal']['temperatures']
heat_capacity = result['thermal']['heat_capacity']

# Heat capacity at 300 K
idx_300K = (temperatures >= 300).argmax()
Cv_300K = heat_capacity[idx_300K]
print(f"Heat capacity at 300 K: {Cv_300K:.2f} J/(K·mol)")
```

**CLI equivalent:**
```bash
orb-infer phonon structure.cif --supercell 2,2,2 --mesh 20,20,20
```

### 4. Bulk Modulus

```python
# Calculate bulk modulus via EOS
result = orb.bulk_modulus(
    atoms,
    strain_range=0.05,    # ±5% volume strain
    n_points=7,           # 7 volume points
    optimize_first=True   # Optimize before EOS
)

B = result['bulk_modulus']
print(f"Bulk modulus: {B:.2f} GPa")
```

**CLI equivalent:**
```bash
orb-infer bulk-modulus structure.cif --strain 0.05 --points 7
```

### 5. Adsorption Energy

```python
# Calculate adsorption energy
result = orb.adsorption_energy(
    host='mof.cif',           # MOF structure
    guest='co2.xyz',          # CO2 molecule
    complex_atoms='mof_co2.cif',  # MOF + CO2
    optimize_complex=True
)

E_ads = result['E_ads']
print(f"Adsorption energy: {E_ads:.3f} eV")
if E_ads < 0:
    print("→ Stable adsorption")
```

**CLI equivalent:**
```bash
orb-infer adsorption --host mof.cif --guest co2.xyz --complex mof_co2.cif
```

## Model Selection

Choose the right model for your task:

```python
# General use (RECOMMENDED)
orb = OrbInference(model_name='orb-v3-omat', device='cuda')

# Materials Project focus
orb = OrbInference(model_name='orb-v3-mpa', device='cuda')

# Dispersion-critical systems (v2 with D3)
orb = OrbInference(model_name='orb-d3-v2', device='cuda')

# CPU-only mode
orb = OrbInference(model_name='orb-v3-omat', device='cpu')
```

**Model Comparison:**

| Model | Training Data | Conservative Forces | Neighbor Limit | Dispersion |
|-------|---------------|---------------------|----------------|------------|
| orb-v3-omat | OMAT24 | ✓ | None | - |
| orb-v3-mpa | MPtraj+Alexandria | ✓ | None | - |
| orb-d3-v2 | MPtraj | ✗ | 30 | Built-in D3 |

## Device Management

```python
from orb_inference.utils.device import get_device, get_device_info

# Auto-detect best device
device = get_device()
print(f"Using device: {device}")

# Get device info
info = get_device_info(device)
print(f"GPU: {info['name']}")
print(f"Memory: {info['memory_allocated']:.2f} / {info['memory_total']:.2f} GB")
```

## Working with Files

```python
from orb_inference.utils.io import load_structure, save_structure

# Load structure (auto-detects format)
atoms = load_structure('structure.cif')  # or .vasp, .xyz, .pdb

# Save structure
save_structure(atoms, 'output.cif')
save_structure(atoms, 'POSCAR', format='vasp')

# Parse input (accepts Atoms object or filepath)
from orb_inference.utils.io import parse_structure_input
atoms = parse_structure_input('structure.cif')  # or Atoms object
```

## Next Steps

1. **Run Examples**: Try all 5 examples in [`examples/`](examples/)
2. **Read Task Guide**: See [Orb_inference_tasks.md](Orb_inference_tasks.md) for 8 task categories
3. **API Reference**: Check [Orb_inference_API_reference.md](Orb_inference_API_reference.md)
4. **Troubleshooting**: See [INSTALL.md](INSTALL.md#troubleshooting)

## Common Pitfalls

1. **PyTorch 2.4.1**: Use PyTorch 2.3.1 instead (compatibility issues)
   ```bash
   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
   ```

2. **FrechetCellFilter error**: Upgrade ASE to ≥ 3.23.0
   ```bash
   pip install --upgrade ase>=3.23.0
   ```

3. **CUDA out of memory**: Reduce precision or batch size
   ```python
   orb = OrbInference(model_name='orb-v3-omat', precision='float32-high')
   ```

4. **Phonon calculation slow**: Use smaller supercell or fewer displacements
   ```python
   result = orb.phonon(atoms, supercell_matrix=[2,2,2])  # instead of [3,3,3]
   ```

## Getting Help

- **Documentation**: [Full task guide](Orb_inference_tasks.md)
- **Issues**: [GitHub Issues](https://github.com/MLFF-inference/orb-inference/issues)
- **Examples**: [`examples/`](examples/) directory
