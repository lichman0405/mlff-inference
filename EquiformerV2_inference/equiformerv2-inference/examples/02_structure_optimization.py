"""
Example 2: Structure Optimization

This example demonstrates:
- Structure optimization (atomic positions)
- Cell optimization (lattice parameters)
- Combined position + cell optimization
- Monitoring convergence
- Using different optimizers

Requirements:
- equiformerv2-inference
- ASE (Atomic Simulation Environment)
"""

from ase.build import bulk
from equiformerv2_inference import EquiformerV2Inference
import numpy as np

print("=" * 60)
print("EquiformerV2 Inference - Example 02: Structure Optimization")
print("=" * 60)

# ====================================
# 1. Initialize EquiformerV2 Model
# ====================================
print("\n1. Initializing EquiformerV2 model...")
calc = EquiformerV2Inference(
    model="equiformer_v2_31M",
    device="auto"
)
print("✓ Model initialized")

# ====================================
# 2. Create and Perturb Structure
# ====================================
print("\n2. Creating and perturbing Cu FCC structure...")
atoms = bulk('Cu', 'fcc', a=3.6)
atoms = atoms * (2, 2, 2)  # 32 atoms

# Save original for comparison
atoms_original = atoms.copy()

# Perturb structure
np.random.seed(42)
atoms.rattle(stdev=0.1)  # Displace atoms randomly by ~0.1 Å
atoms.set_cell(atoms.get_cell() * 1.03, scale_atoms=True)  # Scale cell by 3%

print(f"✓ Created structure: {len(atoms)} atoms")
print(f"  Original volume: {atoms_original.get_volume():.2f} Å³")
print(f"  Perturbed volume: {atoms.get_volume():.2f} Å³")

# Calculate initial energy
result_initial = calc.single_point(atoms)
print(f"\nInitial state:")
print(f"  Energy: {result_initial['energy']:.6f} eV")
print(f"  Max force: {result_initial['max_force']:.6f} eV/Å")
print(f"  Pressure: {result_initial['pressure_GPa']:.4f} GPa")

# ====================================
# 3. Optimize Atomic Positions Only
# ====================================
print("\n3. Optimizing atomic positions (fixed cell)...")
opt_result = calc.optimize(
    atoms.copy(),
    fmax=0.01,           # Convergence criterion: max force < 0.01 eV/Å
    steps=200,           # Maximum optimization steps
    optimizer='LBFGS',   # Optimizer: LBFGS, BFGS, FIRE
    optimize_cell=False, # Keep cell fixed
    logfile='opt_positions.log'
)

atoms_opt_pos = opt_result['atoms']
print("✓ Position optimization completed")
print(f"  Converged: {opt_result['converged']}")
print(f"  Steps taken: {opt_result['steps']}")
print(f"  Initial energy: {opt_result['initial_energy']:.6f} eV")
print(f"  Final energy: {opt_result['final_energy']:.6f} eV")
print(f"  Energy change: {opt_result['final_energy'] - opt_result['initial_energy']:.6f} eV")

result_pos = calc.single_point(atoms_opt_pos)
print(f"  Final max force: {result_pos['max_force']:.6f} eV/Å")
print(f"  Final volume: {atoms_opt_pos.get_volume():.2f} Å³")

# ====================================
# 4. Optimize Cell Only
# ====================================
print("\n4. Optimizing cell parameters (fixed positions)...")

# Start fresh with perturbed structure
atoms_cell = atoms.copy()
opt_result_cell = calc.optimize(
    atoms_cell,
    fmax=0.05,           # Looser convergence for cell-only
    steps=100,
    optimizer='LBFGS',
    optimize_cell=True,
    optimize_positions=False,  # Fix positions, optimize cell only
    pressure_GPa=0.0,    # Target pressure for cell optimization
    logfile='opt_cell.log'
)

atoms_opt_cell = opt_result_cell['atoms']
print("✓ Cell optimization completed")
print(f"  Converged: {opt_result_cell['converged']}")
print(f"  Steps taken: {opt_result_cell['steps']}")
print(f"  Initial volume: {atoms.get_volume():.2f} Å³")
print(f"  Final volume: {atoms_opt_cell.get_volume():.2f} Å³")
print(f"  Volume change: {(atoms_opt_cell.get_volume() - atoms.get_volume()) / atoms.get_volume() * 100:.2f}%")

result_cell = calc.single_point(atoms_opt_cell)
print(f"  Final pressure: {result_cell['pressure_GPa']:.4f} GPa")

# ====================================
# 5. Full Optimization (Positions + Cell)
# ====================================
print("\n5. Full optimization (positions + cell)...")

# Start fresh with perturbed structure
atoms_full = atoms.copy()
opt_result_full = calc.optimize(
    atoms_full,
    fmax=0.01,
    steps=300,
    optimizer='LBFGS',
    optimize_cell=True,
    optimize_positions=True,  # Optimize both positions and cell
    pressure_GPa=0.0,
    logfile='opt_full.log'
)

atoms_opt_full = opt_result_full['atoms']
print("✓ Full optimization completed")
print(f"  Converged: {opt_result_full['converged']}")
print(f"  Steps taken: {opt_result_full['steps']}")
print(f"  Initial energy: {opt_result_full['initial_energy']:.6f} eV")
print(f"  Final energy: {opt_result_full['final_energy']:.6f} eV")
print(f"  Energy change: {opt_result_full['final_energy'] - opt_result_full['initial_energy']:.6f} eV")

result_full = calc.single_point(atoms_opt_full)
print(f"  Final max force: {result_full['max_force']:.6f} eV/Å")
print(f"  Final pressure: {result_full['pressure_GPa']:.4f} GPa")
print(f"  Final volume: {atoms_opt_full.get_volume():.2f} Å³")

# ====================================
# 6. Compare Optimizers
# ====================================
print("\n6. Comparing different optimizers...")

optimizers = ['BFGS', 'LBFGS', 'FIRE']
results_comparison = {}

for opt_name in optimizers:
    atoms_test = atoms.copy()
    opt_result_test = calc.optimize(
        atoms_test,
        fmax=0.01,
        steps=200,
        optimizer=opt_name,
        optimize_cell=False,
        logfile=None  # Suppress log files
    )
    results_comparison[opt_name] = opt_result_test

print("\nOptimizer comparison (positions only):")
print(f"{'Optimizer':<10} {'Steps':<8} {'Converged':<12} {'Final Energy (eV)':<20}")
print("-" * 60)
for opt_name, res in results_comparison.items():
    print(f"{opt_name:<10} {res['steps']:<8} {str(res['converged']):<12} {res['final_energy']:<20.6f}")

# ====================================
# 7. Save Optimized Structure
# ====================================
print("\n7. Saving optimized structure...")
atoms_opt_full.write('optimized_structure.cif')
print("✓ Optimized structure saved to optimized_structure.cif")

# ====================================
# 8. Summary
# ====================================
print("\n" + "=" * 60)
print("Summary:")
print(f"  Perturbed structure energy: {result_initial['energy']:.6f} eV")
print(f"  Position-optimized energy:  {opt_result['final_energy']:.6f} eV")
print(f"  Cell-optimized energy:      {opt_result_cell['final_energy']:.6f} eV")
print(f"  Fully-optimized energy:     {opt_result_full['final_energy']:.6f} eV")
print(f"\n  Volume change (full opt):   {atoms.get_volume():.2f} → {atoms_opt_full.get_volume():.2f} Å³")
print(f"  Pressure (full opt):        {result_full['pressure_GPa']:.4f} GPa")
print(f"  Max force (full opt):       {result_full['max_force']:.6f} eV/Å")
print("=" * 60)
