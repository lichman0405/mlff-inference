"""
Example 1: Basic Usage of Orb Inference

This example demonstrates:
1. Model initialization
2. Single-point energy calculation
3. Structure optimization
4. Saving results
"""

from orb_inference import OrbInference
from orb_inference.utils.io import load_structure, save_structure
from ase.build import bulk

# Create example structure (Cu FCC)
atoms = bulk('Cu', 'fcc', a=3.6)
atoms = atoms.repeat((2, 2, 2))

print("="*60)
print("Orb Inference - Basic Usage Example")
print("="*60)

# Initialize Orb with v3 model (conservative forces)
print("\n1. Initializing Orb model...")
orb = OrbInference(
    model_name='orb-v3-omat',  # OMAT24 dataset, recommended
    device='cuda',              # Use GPU if available
    precision='float32-high'    # Fast and accurate
)

# Get model information
info = orb.info()
print(f"   Model: {info['model_name']}")
print(f"   Device: {info['device']}")
print(f"   Precision: {info['precision']}")

# Single-point energy calculation
print("\n2. Calculating single-point energy...")
result = orb.single_point(atoms)

print(f"   Energy: {result['energy']:.6f} eV")
print(f"   Energy per atom: {result['energy']/len(atoms):.6f} eV/atom")
print(f"   Max force: {result['max_force']:.6f} eV/Å")
print(f"   RMS force: {result['rms_force']:.6f} eV/Å")
print(f"   Pressure: {result['pressure']:.4f} GPa")

# Structure optimization
print("\n3. Optimizing structure...")
opt_result = orb.optimize(
    atoms,
    fmax=0.01,           # Force convergence criterion
    optimizer='LBFGS',   # Optimizer type
    relax_cell=True,     # Relax both atoms and cell
    output='optimization.traj'
)

print(f"   Converged: {opt_result['converged']}")
print(f"   Steps: {opt_result['steps']}")
print(f"   Initial energy: {opt_result['initial_energy']:.6f} eV")
print(f"   Final energy: {opt_result['final_energy']:.6f} eV")
print(f"   Energy change: {opt_result['final_energy'] - opt_result['initial_energy']:.6f} eV")
print(f"   Final fmax: {opt_result['final_fmax']:.6f} eV/Å")

# Get optimized structure
optimized_atoms = opt_result['atoms']
print(f"\n   Optimized cell parameters:")
print(f"   a = {optimized_atoms.cell[0, 0]:.4f} Å")
print(f"   Volume = {optimized_atoms.get_volume():.4f} Å³")

# Save optimized structure
save_structure(optimized_atoms, 'optimized_cu.cif')
print("\n4. Optimized structure saved to 'optimized_cu.cif'")

# Calculate forces on optimized structure
print("\n5. Verifying optimization (forces on optimized structure)...")
final_result = orb.single_point(optimized_atoms)
print(f"   Max force: {final_result['max_force']:.6f} eV/Å (should be < {opt_result['fmax']})")
print(f"   RMS force: {final_result['rms_force']:.6f} eV/Å")

print("\n" + "="*60)
print("Example completed successfully!")
print("="*60)
