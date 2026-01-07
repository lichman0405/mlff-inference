# GRACE Inference - Quick Start Guide

## Installation

```bash
# Basic installation (CPU)
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# Full installation (all features)
pip install -e ".[all]"
```

## Basic Usage

### Python API

```python
from grace_inference import GRACEInference

# Initialize calculator
calc = GRACEInference(model="mof-v1", device="auto")

# Single-point energy
result = calc.single_point("mof.cif")
print(f"Energy: {result['energy']:.6f} eV")

# Structure optimization
optimized = calc.optimize("mof.cif", fmax=0.05, output="optimized.cif")

# Molecular dynamics
final = calc.run_md("mof.cif", ensemble="nvt", temperature_K=300, steps=1000)

# Phonon calculation
phonon = calc.phonon("mof.cif", supercell_matrix=[2, 2, 2])

# Bulk modulus
bm = calc.bulk_modulus("mof.cif")
print(f"Bulk Modulus: {bm['B_GPa']:.2f} GPa")

# Adsorption energy
E_ads = calc.adsorption_energy(
    mof_structure="mof.cif",
    gas_molecule="CO2",
    site_position=[10.0, 10.0, 10.0]
)
print(f"E_ads = {E_ads['E_ads']:.3f} eV")

# Coordination analysis
coord = calc.analyze_coordination("mof.cif", center_element="Zn")
print(f"Average coordination: {coord['avg_coordination']:.2f}")
```

### Command Line Interface

```bash
# Single-point energy
grace-infer energy mof.cif --model mof-v1

# Structure optimization
grace-infer optimize mof.cif --fmax 0.05 --output optimized.cif

# Molecular dynamics
grace-infer md mof.cif --ensemble nvt --temp 300 --steps 10000 --trajectory md.traj

# Phonon calculation
grace-infer phonon mof.cif --supercell 2 2 2 --temp-range 0 1000 10

# Bulk modulus
grace-infer bulk-modulus mof.cif

# Adsorption energy
grace-infer adsorption mof.cif --gas CO2 --site 10.0 10.0 10.0

# Coordination analysis
grace-infer coordination mof.cif --center Zn --cutoff 3.0

# Show system info
grace-infer info --verbose
```

## Examples

### Example 1: Single-Point Energy Calculation

```python
from ase.io import read
from grace_inference import GRACEInference

# Load structure
atoms = read("IRMOF-1.cif")

# Initialize calculator
calc = GRACEInference(model="mof-v1", device="cuda")

# Calculate energy and forces
result = calc.single_point(atoms)

print(f"Energy: {result['energy']:.6f} eV")
print(f"Forces (max): {result['forces'].max():.4f} eV/Å")
print(f"Forces (RMS): {(result['forces']**2).mean()**0.5:.4f} eV/Å")
```

### Example 2: Structure Optimization

```python
from grace_inference import GRACEInference

calc = GRACEInference(model="mof-v1")

# Optimize structure
optimized = calc.optimize(
    structure="mof.cif",
    fmax=0.05,           # Force convergence criterion
    max_steps=500,       # Maximum optimization steps
    optimizer="BFGS",    # Optimization algorithm
    output="optimized.cif"
)

print(f"Optimization converged: {optimized.info['converged']}")
print(f"Final energy: {optimized.get_potential_energy():.6f} eV")
print(f"Steps taken: {optimized.info['steps']}")
```

### Example 3: Molecular Dynamics Simulation

```python
from grace_inference import GRACEInference

calc = GRACEInference(model="mof-v1")

# Run NVT molecular dynamics
final_atoms = calc.run_md(
    structure="mof.cif",
    ensemble="nvt",           # NVT ensemble
    temperature_K=300,        # Temperature in Kelvin
    steps=10000,              # Number of MD steps
    timestep_fs=1.0,          # Timestep in fs
    trajectory="md.traj",     # Output trajectory file
    log_interval=100          # Log every 100 steps
)

print(f"Simulation completed")
print(f"Final temperature: {final_atoms.get_temperature():.2f} K")
```

### Example 4: Phonon Calculation

```python
from grace_inference import GRACEInference

calc = GRACEInference(model="mof-v1")

# Calculate phonon properties
result = calc.phonon(
    structure="mof.cif",
    supercell_matrix=[2, 2, 2],  # 2×2×2 supercell
    temp_range=[0, 1000, 10],    # Temperature range: 0-1000K, step 10K
    save_band_yaml="phonon_band.yaml",
    save_dos_dat="phonon_dos.dat"
)

print(f"Zero-point energy: {result['ZPE']:.4f} eV")
print(f"Heat capacity (300K): {result['Cv_300K']:.4f} J/mol/K")
print(f"Entropy (300K): {result['S_300K']:.4f} J/mol/K")
```

### Example 5: Gas Adsorption Energy

```python
from grace_inference import GRACEInference

calc = GRACEInference(model="mof-v1")

# Calculate CO2 adsorption energy
result = calc.adsorption_energy(
    mof_structure="IRMOF-1.cif",
    gas_molecule="CO2",               # Gas molecule
    site_position=[10.0, 10.0, 10.0], # Adsorption site (Å)
    relax_mof=True,                   # Relax MOF structure
    relax_complex=True                # Relax MOF+gas complex
)

print(f"Adsorption energy: {result['E_ads']:.3f} eV")
print(f"Gas-framework distance: {result['distance']:.3f} Å")
print(f"MOF energy: {result['E_mof']:.6f} eV")
print(f"Gas energy: {result['E_gas']:.6f} eV")
print(f"Complex energy: {result['E_complex']:.6f} eV")
```

### Example 6: Coordination Environment Analysis

```python
from grace_inference import GRACEInference

calc = GRACEInference(model="mof-v1")

# Analyze Zn coordination in MOF-5
result = calc.analyze_coordination(
    structure="MOF-5.cif",
    center_element="Zn",      # Metal center
    ligand_element="O",       # Coordinating atom
    cutoff_radius=3.0         # Cutoff distance (Å)
)

print(f"Number of Zn atoms: {result['n_centers']}")
print(f"Average coordination: {result['avg_coordination']:.2f}")
print(f"Coordination distribution: {result['coordination_distribution']}")
print(f"Average Zn-O distance: {result['avg_distance']:.3f} Å")
```

### Example 7: Bulk Modulus Calculation

```python
from grace_inference import GRACEInference

calc = GRACEInference(model="mof-v1")

# Calculate bulk modulus
result = calc.bulk_modulus(
    structure="mof.cif",
    strain_range=0.05,    # ±5% strain
    n_points=7            # 7 data points
)

print(f"Bulk modulus: {result['B_GPa']:.2f} GPa")
print(f"Equilibrium volume: {result['V0']:.3f} Å³")
print(f"Equilibrium energy: {result['E0']:.6f} eV")
```

### Example 8: High-Throughput Screening

```python
from grace_inference import GRACEInference
import glob
from tqdm import tqdm
import json

calc = GRACEInference(model="mof-v1")

# Screen MOF database for H2 adsorption
mof_files = glob.glob("database/*.cif")
results = []

for mof_file in tqdm(mof_files, desc="Screening MOFs"):
    try:
        # Calculate adsorption energy
        E_ads = calc.adsorption_energy(
            mof_structure=mof_file,
            gas_molecule="H2",
            site_position="auto",  # Auto-detect best site
            relax_complex=True
        )
        
        results.append({
            'name': mof_file,
            'E_ads': E_ads['E_ads'],
            'distance': E_ads['distance']
        })
    except Exception as e:
        print(f"Error processing {mof_file}: {e}")

# Sort by adsorption energy
results.sort(key=lambda x: x['E_ads'])

# Save results
with open("screening_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Print top 10
print("\nTop 10 MOFs for H2 adsorption:")
for i, r in enumerate(results[:10], 1):
    print(f"{i:2d}. {r['name']:40s} E_ads = {r['E_ads']:7.3f} eV")
```

## Model Selection

GRACE provides different pre-trained models:

```python
# General purpose MOF model
calc = GRACEInference(model="mof-v1")

# High-accuracy model (slower)
calc = GRACEInference(model="mof-v2-large")

# Fast screening model (less accurate)
calc = GRACEInference(model="mof-v1-fast")

# Custom model from checkpoint
calc = GRACEInference(model_path="/path/to/checkpoint.pt")
```

## Device Management

```python
# Automatic device selection (recommended)
calc = GRACEInference(device="auto")

# Force CPU
calc = GRACEInference(device="cpu")

# Force CUDA GPU
calc = GRACEInference(device="cuda")

# Specific GPU
calc = GRACEInference(device="cuda:0")

# Check device info
from grace_inference.utils import get_device_info
info = get_device_info()
print(f"Using device: {info['device']}")
print(f"CUDA available: {info['cuda_available']}")
```

## Troubleshooting

### Import Error

```bash
# Check installation
python -c "import grace_inference; print(grace_inference.__version__)"

# Reinstall if needed
pip install -e . --force-reinstall
```

### GPU Not Detected

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Check DGL CUDA support
import dgl
print(f"DGL CUDA enabled: {dgl.cuda.is_available()}")
```

### Memory Issues

```python
# Use smaller batch size for large structures
calc = GRACEInference(model="mof-v1", batch_size=1)

# Clear cache
import torch
torch.cuda.empty_cache()
```

## Next Steps

- Read the full [Installation Guide](INSTALL_GUIDE.md)
- Explore more [Examples](examples/)
- Check the [API Reference](docs/API_REFERENCE.md)
- Join the community discussions

---

For more information, visit the [GRACE Inference GitHub repository](https://github.com/yourusername/grace-inference).
