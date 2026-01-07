"""
Example 1: Basic Usage - Single-point energy and structure optimization

This example demonstrates the basic functionality of mace-inference:
- Loading a structure
- Calculating single-point energy
- Structure optimization
"""

from ase.build import bulk
from mace_inference import MACEInference

# Create MACE calculator
print("Initializing MACE calculator...")
calc = MACEInference(model="medium", device="auto")

# Create a simple structure (Cu FCC)
print("\nCreating structure...")
atoms = bulk('Cu', 'fcc', a=3.6)
atoms = atoms * (2, 2, 2)  # Create 2x2x2 supercell

print(f"Number of atoms: {len(atoms)}")

# Single-point energy calculation
print("\n=== Single-Point Energy ===")
result = calc.single_point(atoms)

print(f"Total Energy:    {result['energy']:.6f} eV")
print(f"Energy/atom:     {result['energy_per_atom']:.6f} eV")
print(f"Max Force:       {result['max_force']:.6f} eV/Å")
print(f"RMS Force:       {result['rms_force']:.6f} eV/Å")
print(f"Pressure:        {result['pressure_GPa']:.4f} GPa")

# Perturb structure slightly
print("\n=== Perturbing Structure ===")
import numpy as np
np.random.seed(42)
atoms.rattle(stdev=0.1)  # Random displacement

result_perturbed = calc.single_point(atoms)
print(f"Energy after perturbation: {result_perturbed['energy']:.6f} eV")
print(f"Max Force after perturbation: {result_perturbed['max_force']:.6f} eV/Å")

# Optimize structure
print("\n=== Structure Optimization ===")
optimized = calc.optimize(
    atoms,
    fmax=0.01,
    steps=200,
    optimizer="LBFGS",
    logfile="optimization.log"
)

result_opt = calc.single_point(optimized)
print(f"Energy after optimization: {result_opt['energy']:.6f} eV")
print(f"Max Force after optimization: {result_opt['max_force']:.6f} eV/Å")

# Save optimized structure
optimized.write("optimized_Cu.cif")
print("\n✓ Optimized structure saved to optimized_Cu.cif")

print("\n=== Summary ===")
print(f"Energy change: {result_opt['energy'] - result_perturbed['energy']:.6f} eV")
print(f"Force reduction: {result_perturbed['max_force']} → {result_opt['max_force']:.6f} eV/Å")
