"""
Example 5: High-Throughput Screening

This example demonstrates batch processing of multiple structures.
"""

from ase.build import bulk
from mace_inference import MACEInference
import time

# Initialize calculator once
print("Initializing MACE calculator...")
calc = MACEInference(model="medium", device="auto")

# Create a set of structures to screen
structures_to_screen = {
    "Cu_fcc": bulk('Cu', 'fcc', a=3.6),
    "Al_fcc": bulk('Al', 'fcc', a=4.05),
    "Fe_bcc": bulk('Fe', 'bcc', a=2.87),
    "Si_diamond": bulk('Si', 'diamond', a=5.43),
    "NaCl": bulk('NaCl', 'rocksalt', a=5.64),
}

print(f"\nScreening {len(structures_to_screen)} structures...")

results = {}

# Batch single-point calculations
print("\n=== Single-Point Energy Calculations ===")
start_time = time.time()

for name, atoms in structures_to_screen.items():
    result = calc.single_point(atoms)
    results[name] = {
        'energy_per_atom': result['energy_per_atom'],
        'max_force': result['max_force'],
        'pressure_GPa': result['pressure_GPa']
    }
    print(f"{name:<15} E/atom: {result['energy_per_atom']:>10.6f} eV  "
          f"P: {result['pressure_GPa']:>8.4f} GPa")

elapsed = time.time() - start_time
print(f"\nTotal time: {elapsed:.2f} s ({elapsed/len(structures_to_screen):.2f} s per structure)")

# Batch structure optimization
print("\n=== Structure Optimization ===")
optimized_structures = {}

for name, atoms in structures_to_screen.items():
    print(f"\nOptimizing {name}...")
    optimized = calc.optimize(
        atoms,
        fmax=0.01,
        steps=200,
        optimize_cell=True
    )
    
    # Calculate properties of optimized structure
    result_opt = calc.single_point(optimized)
    
    optimized_structures[name] = optimized
    results[name]['optimized_energy'] = result_opt['energy_per_atom']
    results[name]['optimized_volume'] = optimized.get_volume()
    
    print(f"  Energy/atom: {result_opt['energy_per_atom']:.6f} eV")
    print(f"  Volume: {optimized.get_volume():.2f} Å³")

# Calculate bulk modulus for selected structures
print("\n=== Bulk Modulus Calculations ===")
for name in ["Cu_fcc", "Al_fcc", "Si_diamond"]:
    print(f"\nCalculating bulk modulus for {name}...")
    atoms = optimized_structures[name]
    
    bm_result = calc.bulk_modulus(atoms, n_points=7)
    results[name]['bulk_modulus_GPa'] = bm_result['B_GPa']
    
    print(f"  B = {bm_result['B_GPa']:.2f} GPa")

# Export results
print("\n=== Exporting Results ===")
import json

with open('screening_results.json', 'w') as f:
    # Convert numpy types to native Python types for JSON
    export_data = {}
    for name, data in results.items():
        export_data[name] = {
            k: float(v) if hasattr(v, 'item') else v 
            for k, v in data.items()
        }
    json.dump(export_data, f, indent=2)

print("✓ Results saved to screening_results.json")

# Save optimized structures
for name, atoms in optimized_structures.items():
    filename = f"optimized_{name}.cif"
    atoms.write(filename)
    print(f"✓ Saved {filename}")

print("\n=== Summary ===")
print(f"Screened {len(structures_to_screen)} structures")
print(f"Average time per structure: {elapsed/len(structures_to_screen):.2f} s")
