"""
Example 1: Single-Point Energy Calculation

This example demonstrates:
- Initializing EquiformerV2 model
- Single-point energy calculation
- Accessing energy, forces, and stress
- Basic property calculations

Requirements:
- equiformerv2-inference
- ASE (Atomic Simulation Environment)
"""

from ase.build import bulk, molecule
from equiformerv2_inference import EquiformerV2Inference
import numpy as np

print("=" * 60)
print("EquiformerV2 Inference - Example 01: Single-Point Energy")
print("=" * 60)

# ====================================
# 1. Initialize EquiformerV2 Model
# ====================================
print("\n1. Initializing EquiformerV2 model...")
calc = EquiformerV2Inference(
    model="equiformer_v2_31M",  # Available models: equiformer_v2_31M, equiformer_v2_153M
    device="auto"                # Use 'auto' for automatic GPU/CPU selection
)
print(f"✓ Model initialized")
print(f"  Device: {calc.device}")

# ====================================
# 2. Create Test Structure (Crystal)
# ====================================
print("\n2. Creating test structure (Cu FCC crystal)...")
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
atoms = atoms * (2, 2, 2)  # Create 2x2x2 supercell (32 atoms)

print(f"✓ Created structure:")
print(f"  Formula: {atoms.get_chemical_formula()}")
print(f"  Number of atoms: {len(atoms)}")
print(f"  Volume: {atoms.get_volume():.2f} Å³")
print(f"  Cell parameters: {atoms.cell.cellpar()}")

# ====================================
# 3. Single-Point Energy Calculation
# ====================================
print("\n3. Running single-point energy calculation...")
result = calc.single_point(atoms)

print("✓ Calculation completed")
print(f"\n  Total Energy:      {result['energy']:.6f} eV")
print(f"  Energy per atom:   {result['energy_per_atom']:.6f} eV/atom")
print(f"  Max force:         {result['max_force']:.6f} eV/Å")
print(f"  RMS force:         {result['rms_force']:.6f} eV/Å")
print(f"  Pressure:          {result['pressure_GPa']:.4f} GPa")

# ====================================
# 4. Access Detailed Properties
# ====================================
print("\n4. Detailed force and stress information:")

# Access forces
forces = result['forces']
print(f"\n  Forces shape: {forces.shape}")
print(f"  Forces on first 3 atoms (eV/Å):")
for i in range(min(3, len(atoms))):
    fx, fy, fz = forces[i]
    f_mag = np.linalg.norm(forces[i])
    print(f"    Atom {i}: [{fx:8.5f}, {fy:8.5f}, {fz:8.5f}]  |F| = {f_mag:.5f}")

# Access stress
if 'stress' in result:
    stress = result['stress']
    print(f"\n  Stress tensor (GPa):")
    stress_gpa = stress * 160.21766208  # Convert eV/Å³ to GPa
    print(f"    σxx = {stress_gpa[0]:8.4f}")
    print(f"    σyy = {stress_gpa[1]:8.4f}")
    print(f"    σzz = {stress_gpa[2]:8.4f}")
    print(f"    σyz = {stress_gpa[3]:8.4f}")
    print(f"    σxz = {stress_gpa[4]:8.4f}")
    print(f"    σxy = {stress_gpa[5]:8.4f}")

# ====================================
# 5. Molecule Example
# ====================================
print("\n5. Single-point calculation for a molecule (H2O)...")
water = molecule('H2O')
water.center(vacuum=10.0)  # Add vacuum for isolated molecule

result_mol = calc.single_point(water)

print(f"✓ Water molecule calculation:")
print(f"  Total Energy:      {result_mol['energy']:.6f} eV")
print(f"  Energy per atom:   {result_mol['energy_per_atom']:.6f} eV/atom")
print(f"  Max force:         {result_mol['max_force']:.6f} eV/Å")

# ====================================
# 6. Perturbed Structure
# ====================================
print("\n6. Comparing energy of perturbed structure...")
atoms_perturbed = atoms.copy()
np.random.seed(42)
atoms_perturbed.rattle(stdev=0.05)  # Random displacement of 0.05 Å

result_perturbed = calc.single_point(atoms_perturbed)

print(f"  Original energy:   {result['energy']:.6f} eV")
print(f"  Perturbed energy:  {result_perturbed['energy']:.6f} eV")
print(f"  Energy difference: {result_perturbed['energy'] - result['energy']:.6f} eV")
print(f"  Max force (original):  {result['max_force']:.6f} eV/Å")
print(f"  Max force (perturbed): {result_perturbed['max_force']:.6f} eV/Å")

# ====================================
# 7. Summary
# ====================================
print("\n" + "=" * 60)
print("Summary:")
print("  ✓ Initialized EquiformerV2 model")
print("  ✓ Calculated energy for Cu crystal (32 atoms)")
print("  ✓ Calculated energy for H2O molecule")
print("  ✓ Analyzed forces and stress")
print("  ✓ Compared perturbed structure")
print("=" * 60)
