"""
Example 1: Basic Usage of eSEN Inference

This example demonstrates:
- Initializing eSEN model
- Single-point energy calculation
- Structure optimization
- Accessing results

Requirements:
- esen-inference
- Example data: MOF-5.cif
"""

from esen_inference import ESENInference
from ase.io import read, write
from ase.build import bulk

print("=" * 60)
print("Example 1: Basic Usage of eSEN Inference")
print("=" * 60)

# ====================================
# 1. Initialize eSEN Model
# ====================================
print("\n1. Initializing eSEN model...")
esen = ESENInference(
    model_name='esen-30m-oam',  # MOFSimBench #1 model
    device='cuda',              # Use 'cpu' if no GPU
    precision='float32'
)
print(f"✓ Model initialized: {esen}")

# ====================================
# 2. Create/Load Test Structure
# ====================================
print("\n2. Creating test structure (Cu metal)...")
# Create a simple Cu crystal
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
atoms = atoms * (2, 2, 2)  # 32 atoms
print(f"✓ Created structure: {len(atoms)} atoms")
print(f"  Formula: {atoms.get_chemical_formula()}")
print(f"  Volume: {atoms.get_volume():.2f} Å³")

# ====================================
# 3. Single-Point Calculation
# ====================================
print("\n3. Single-point energy calculation...")
result = esen.single_point(atoms)

print("✓ Results:")
print(f"  Energy: {result['energy']:.6f} eV")
print(f"  Energy per atom: {result['energy_per_atom']:.6f} eV/atom")
print(f"  Max force: {result['max_force']:.6f} eV/Å")
print(f"  RMS force: {result['rms_force']:.6f} eV/Å")
print(f"  Pressure: {result['pressure']:.4f} GPa")

# ====================================
# 4. Structure Optimization
# ====================================
print("\n4. Structure optimization...")
# Slightly perturb structure
import numpy as np
atoms_perturbed = atoms.copy()
atoms_perturbed.positions += np.random.normal(0, 0.05, atoms_perturbed.positions.shape)

print("Optimizing (coordinates only)...")
opt_result = esen.optimize(
    atoms_perturbed,
    fmax=0.01,           # Converge when max force < 0.01 eV/Å
    optimizer='LBFGS',
    relax_cell=False,
    max_steps=100
)

print("✓ Optimization results:")
print(f"  Converged: {opt_result['converged']}")
print(f"  Steps: {opt_result['steps']}")
print(f"  Initial energy: {opt_result['initial_energy']:.6f} eV")
print(f"  Final energy: {opt_result['final_energy']:.6f} eV")
print(f"  Energy降低: {opt_result['energy_change']:.6f} eV")
print(f"  Final fmax: {opt_result['final_fmax']:.6f} eV/Å")

# ====================================
# 5. Full Optimization (Cell + Coords)
# ====================================
print("\n5. Full optimization (cell + coordinates)...")
opt_full = esen.optimize(
    atoms.copy(),
    fmax=0.01,
    optimizer='LBFGS',
    relax_cell=True,    # Also optimize cell
    max_steps=100
)

print("✓ Full optimization results:")
print(f"  Converged: {opt_full['converged']}")
print(f"  Steps: {opt_full['steps']}")
print(f"  Volume change: {(opt_full['atoms'].get_volume() - atoms.get_volume()):.2f} Å³")
print(f"  Final fmax: {opt_full['final_fmax']:.6f} eV/Å")

# ====================================
# 6. Save Results
# ====================================
print("\n6. Saving optimized structure...")
optimized_atoms = opt_full['atoms']
write('Cu_optimized.cif', optimized_atoms)
print("✓ Saved to: Cu_optimized.cif")

# ====================================
# Summary
# ====================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Model: {esen.model_name}")
print(f"Device: {esen.device}")
print(f"Initial energy: {opt_full['initial_energy']:.6f} eV")
print(f"Final energy: {opt_full['final_energy']:.6f} eV")
print(f"Energy gain: {-opt_full['energy_change']:.6f} eV")
print(f"Optimization successful: {opt_full['converged']}")
print("=" * 60)

print("\n✓ Example 1 completed successfully!")
print("\nNext steps:")
print("- Run: python 02_molecular_dynamics.py")
print("- See: python 03_phonon_calculation.py")
