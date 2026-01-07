"""
Example 5: High-Throughput Screening

This example demonstrates:
1. Batch processing of multiple structures
2. Parallel/sequential screening workflow
3. Results aggregation and comparison
"""

from orb_inference import OrbInference
from ase.build import bulk
from ase.io import write
import numpy as np
from pathlib import Path

print("="*60)
print("Orb Inference - High-Throughput Screening Example")
print("="*60)

# Create a small database of structures
print("\n1. Creating test database (5 structures)...")

structures = {
    'Cu_fcc': bulk('Cu', 'fcc', a=3.6),
    'Al_fcc': bulk('Al', 'fcc', a=4.05),
    'Fe_bcc': bulk('Fe', 'bcc', a=2.87),
    'Si_diamond': bulk('Si', 'diamond', a=5.43),
    'C_diamond': bulk('C', 'diamond', a=3.57),
}

# Save structures
data_dir = Path('screening_data')
data_dir.mkdir(exist_ok=True)

for name, atoms in structures.items():
    write(data_dir / f'{name}.cif', atoms)
    print(f"   Created: {name}.cif ({len(atoms)} atoms)")

# Initialize Orb once for all calculations
print("\n2. Initializing Orb model...")
orb = OrbInference(model_name='orb-v3-omat', device='cuda')

# Screening workflow
print("\n3. Running high-throughput screening...")
print("   Tasks: optimization + single-point + bulk modulus")

results = {}

for name, atoms in structures.items():
    print(f"\n   Processing: {name}")
    
    # Initialize result dict
    results[name] = {}
    
    # Optimize structure
    print(f"     - Optimizing...")
    opt_result = orb.optimize(atoms, fmax=0.05, relax_cell=True)
    
    results[name]['optimized_atoms'] = opt_result['atoms']
    results[name]['opt_energy'] = opt_result['final_energy']
    results[name]['opt_steps'] = opt_result['steps']
    results[name]['converged'] = opt_result['converged']
    
    # Single-point on optimized
    print(f"     - Calculating properties...")
    sp_result = orb.single_point(opt_result['atoms'])
    
    results[name]['energy'] = sp_result['energy']
    results[name]['energy_per_atom'] = sp_result['energy'] / len(atoms)
    results[name]['max_force'] = sp_result['max_force']
    results[name]['pressure'] = sp_result['pressure']
    
    # Bulk modulus (quick calculation with fewer points)
    print(f"     - Calculating bulk modulus...")
    bulk_result = orb.bulk_modulus(
        opt_result['atoms'],
        strain_range=0.03,  # Smaller range for speed
        n_points=5,         # Fewer points
        optimize_first=False  # Already optimized
    )
    
    results[name]['bulk_modulus'] = bulk_result['bulk_modulus']
    results[name]['eq_volume'] = bulk_result['equilibrium_volume']
    
    print(f"     ✓ Completed")

# Aggregate and compare results
print("\n" + "="*60)
print("4. Screening Results Summary")
print("="*60)

print(f"\n{'Material':<15} {'E/atom (eV)':<15} {'B (GPa)':<12} {'V (Å³)':<12} {'P (GPa)':<12}")
print("-"*70)

for name in structures.keys():
    r = results[name]
    print(f"{name:<15} {r['energy_per_atom']:>14.6f} {r['bulk_modulus']:>11.2f} "
          f"{r['eq_volume']:>11.3f} {r['pressure']:>11.4f}")

# Find extremes
print(f"\n5. Analysis:")

# Most stable (lowest energy per atom)
most_stable = min(results.items(), key=lambda x: x[1]['energy_per_atom'])
print(f"\n   Most stable (lowest E/atom): {most_stable[0]}")
print(f"     Energy per atom: {most_stable[1]['energy_per_atom']:.6f} eV/atom")

# Highest bulk modulus (hardest)
hardest = max(results.items(), key=lambda x: x[1]['bulk_modulus'])
print(f"\n   Hardest (highest bulk modulus): {hardest[0]}")
print(f"     Bulk modulus: {hardest[1]['bulk_modulus']:.2f} GPa")

# Lowest bulk modulus (most compressible)
softest = min(results.items(), key=lambda x: x[1]['bulk_modulus'])
print(f"\n   Most compressible (lowest bulk modulus): {softest[0]}")
print(f"     Bulk modulus: {softest[1]['bulk_modulus']:.2f} GPa")

# Optimization difficulty (most steps)
hardest_opt = max(results.items(), key=lambda x: x[1]['opt_steps'])
print(f"\n   Most challenging optimization: {hardest_opt[0]}")
print(f"     Optimization steps: {hardest_opt[1]['opt_steps']}")

# Save results to file
print("\n6. Saving results...")

results_file = data_dir / 'screening_results.txt'
with open(results_file, 'w') as f:
    f.write("High-Throughput Screening Results\n")
    f.write("="*70 + "\n\n")
    f.write(f"{'Material':<15} {'E/atom (eV)':<15} {'B (GPa)':<12} {'V (Å³)':<12} {'P (GPa)':<12}\n")
    f.write("-"*70 + "\n")
    
    for name in structures.keys():
        r = results[name]
        f.write(f"{name:<15} {r['energy_per_atom']:>14.6f} {r['bulk_modulus']:>11.2f} "
                f"{r['eq_volume']:>11.3f} {r['pressure']:>11.4f}\n")

print(f"   Results saved to: {results_file}")

# Save optimized structures
opt_dir = data_dir / 'optimized'
opt_dir.mkdir(exist_ok=True)

for name, r in results.items():
    write(opt_dir / f'{name}_opt.cif', r['optimized_atoms'])

print(f"   Optimized structures saved to: {opt_dir}/")

print("\n" + "="*60)
print("High-throughput screening completed!")
print(f"Total structures processed: {len(structures)}")
print(f"All converged: {all(r['converged'] for r in results.values())}")
print("="*60)

# Performance summary
total_atoms = sum(len(atoms) for atoms in structures.values())
print(f"\nPerformance summary:")
print(f"  Total atoms processed: {total_atoms}")
print(f"  Average optimization steps: {np.mean([r['opt_steps'] for r in results.values()]):.1f}")
print(f"  Structures: {len(structures)}")
