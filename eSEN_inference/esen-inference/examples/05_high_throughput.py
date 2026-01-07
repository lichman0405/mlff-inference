"""
Example 5: High-Throughput Screening and Batch Processing

This example demonstrates:
- Batch structure optimization
- High-throughput property calculation
- Parallel processing
- Results aggregation and analysis
- Screening workflow for materials discovery

Requirements:
- esen-inference
"""

from esen_inference import ESENInference
from ase.build import bulk
from ase.io import read, write
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 60)
print("Example 5: High-Throughput Screening with eSEN")
print("=" * 60)

# ====================================
# 1. Initialize
# ====================================
print("\n1. Initializing eSEN model...")
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# ====================================
# 2. Create Test Dataset
# ====================================
print("\n2. Creating test dataset (metal structures)...")

# Create structures for common metals
metals = [
    ('Cu', 3.6, 'fcc'),
    ('Al', 4.05, 'fcc'),
    ('Ni', 3.52, 'fcc'),
    ('Au', 4.08, 'fcc'),
    ('Ag', 4.09, 'fcc'),
    ('Pt', 3.92, 'fcc'),
    ('Fe', 2.87, 'bcc'),
    ('Mo', 3.15, 'bcc'),
    ('W', 3.16, 'bcc'),
    ('Cr', 2.88, 'bcc'),
]

structures = {}
for symbol, a, structure_type in metals:
    atoms = bulk(symbol, structure_type, a=a, cubic=True) * (2, 2, 2)
    structures[symbol] = atoms
    
print(f"✓ Created {len(structures)} test structures")
print(f"  Metals: {', '.join(structures.keys())}")

# ====================================
# 3. Batch Optimization
# ====================================
print("\n3. Running batch optimization...")
print("   (This may take several minutes...)")

output_dir = Path('optimized_structures')
output_dir.mkdir(exist_ok=True)

optimization_results = {}
start_time = datetime.now()

for i, (symbol, atoms) in enumerate(structures.items(), 1):
    print(f"\n[{i}/{len(structures)}] Optimizing {symbol}...")
    
    try:
        opt_result = esen.optimize(
            atoms,
            fmax=0.01,
            relax_cell=True,
            optimizer='LBFGS',
            max_steps=200
        )
        
        # Save optimized structure
        output_file = output_dir / f'{symbol}_optimized.cif'
        write(str(output_file), opt_result['atoms'])
        
        # Store results
        optimization_results[symbol] = {
            'converged': opt_result['converged'],
            'steps': opt_result['steps'],
            'initial_energy': opt_result['initial_energy'],
            'final_energy': opt_result['final_energy'],
            'energy_change': opt_result['energy_change'],
            'final_fmax': opt_result['final_fmax'],
            'volume': opt_result['atoms'].get_volume(),
            'energy_per_atom': opt_result['final_energy'] / len(opt_result['atoms'])
        }
        
        print(f"  ✓ Converged: {opt_result['converged']}, Steps: {opt_result['steps']}, "
              f"E/atom: {optimization_results[symbol]['energy_per_atom']:.4f} eV")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        optimization_results[symbol] = {'error': str(e)}

end_time = datetime.now()
elapsed = (end_time - start_time).total_seconds()

print(f"\n✓ Batch optimization completed in {elapsed:.1f} seconds")
print(f"  Average time per structure: {elapsed/len(structures):.1f} s")

# ====================================
# 4. Property Calculation
# ====================================
print("\n4. Calculating bulk modulus for all structures...")

bulk_moduli = {}
for i, (symbol, atoms) in enumerate(structures.items(), 1):
    if 'error' in optimization_results.get(symbol, {}):
        continue
        
    print(f"[{i}/{len(structures)}] {symbol}...", end=' ')
    
    try:
        # Load optimized structure
        opt_atoms = read(str(output_dir / f'{symbol}_optimized.cif'))
        
        # Calculate bulk modulus
        result = esen.bulk_modulus(
            opt_atoms,
            strain_range=0.03,
            npoints=7
        )
        
        bulk_moduli[symbol] = result['bulk_modulus']
        print(f"B = {result['bulk_modulus']:.1f} GPa")
        
    except Exception as e:
        print(f"Error: {e}")

print(f"\n✓ Bulk modulus calculated for {len(bulk_moduli)} structures")

# ====================================
# 5. Save Results
# ====================================
print("\n5. Saving results to JSON...")

# Combine all results
all_results = {
    'timestamp': datetime.now().isoformat(),
    'model': esen.model_name,
    'total_structures': len(structures),
    'optimization_results': optimization_results,
    'bulk_moduli': bulk_moduli,
    'statistics': {
        'converged': sum(1 for r in optimization_results.values() 
                        if r.get('converged', False)),
        'failed': sum(1 for r in optimization_results.values() 
                     if 'error' in r),
        'total_time': elapsed
    }
}

with open('screening_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("✓ Results saved to: screening_results.json")

# ====================================
# 6. Data Analysis
# ====================================
print("\n6. Analyzing results...")

# Extract energies and volumes
symbols = []
energies_per_atom = []
volumes_per_atom = []
bulk_mod_values = []

for symbol, result in optimization_results.items():
    if 'error' not in result:
        symbols.append(symbol)
        energies_per_atom.append(result['energy_per_atom'])
        volumes_per_atom.append(result['volume'] / len(structures[symbol]))
        if symbol in bulk_moduli:
            bulk_mod_values.append(bulk_moduli[symbol])

print("✓ Data extraction completed")
print(f"  Valid structures: {len(symbols)}")

# ====================================
# 7. Visualization
# ====================================
print("\n7. Creating visualization...")

fig = plt.figure(figsize=(16, 12))

# Create 2x2 grid
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

# 1. Energy per atom
ax1.bar(range(len(symbols)), energies_per_atom, color='steelblue')
ax1.set_xticks(range(len(symbols)))
ax1.set_xticklabels(symbols, rotation=45)
ax1.set_ylabel('Energy per atom (eV)')
ax1.set_title('Cohesive Energy Comparison')
ax1.grid(True, alpha=0.3, axis='y')

# 2. Volume per atom
ax2.bar(range(len(symbols)), volumes_per_atom, color='coral')
ax2.set_xticks(range(len(symbols)))
ax2.set_xticklabels(symbols, rotation=45)
ax2.set_ylabel('Volume per atom (Å³)')
ax2.set_title('Atomic Volume Comparison')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Bulk modulus
if bulk_mod_values:
    symbols_with_B = [s for s in symbols if s in bulk_moduli]
    B_values = [bulk_moduli[s] for s in symbols_with_B]
    
    ax3.bar(range(len(symbols_with_B)), B_values, color='forestgreen')
    ax3.set_xticks(range(len(symbols_with_B)))
    ax3.set_xticklabels(symbols_with_B, rotation=45)
    ax3.set_ylabel('Bulk Modulus (GPa)')
    ax3.set_title('Mechanical Properties Comparison')
    ax3.grid(True, alpha=0.3, axis='y')

# 4. Optimization convergence
converged_counts = {'Converged': 0, 'Not converged': 0, 'Failed': 0}
for result in optimization_results.values():
    if 'error' in result:
        converged_counts['Failed'] += 1
    elif result.get('converged', False):
        converged_counts['Converged'] += 1
    else:
        converged_counts['Not converged'] += 1

ax4.pie(converged_counts.values(), labels=converged_counts.keys(), 
        autopct='%1.1f%%', startangle=90,
        colors=['#90EE90', '#FFB6C1', '#FFD700'])
ax4.set_title('Optimization Success Rate')

plt.tight_layout()
plt.savefig('screening_analysis.png', dpi=300)
print("✓ Visualization saved to: screening_analysis.png")

# ====================================
# 8. Statistical Summary
# ====================================
print("\n8. Statistical summary...")

# Create summary table
print("\n" + "=" * 80)
print(f"{'Symbol':<8} {'E/atom (eV)':<15} {'V/atom (Å³)':<15} {'B (GPa)':<12} {'Steps':<8}")
print("=" * 80)

for symbol in symbols:
    result = optimization_results[symbol]
    E_per_atom = result['energy_per_atom']
    V_per_atom = result['volume'] / len(structures[symbol])
    B = bulk_moduli.get(symbol, float('nan'))
    steps = result['steps']
    
    print(f"{symbol:<8} {E_per_atom:<15.4f} {V_per_atom:<15.2f} {B:<12.1f} {steps:<8}")

print("=" * 80)

# ====================================
# Summary
# ====================================
print("\n" + "=" * 60)
print("High-Throughput Screening Summary")
print("=" * 60)
print(f"Total structures processed: {len(structures)}")
print(f"Successfully optimized: {all_results['statistics']['converged']}")
print(f"Failed: {all_results['statistics']['failed']}")
print(f"Bulk moduli calculated: {len(bulk_moduli)}")
print(f"Total computation time: {elapsed:.1f} seconds")
print(f"Average time per structure: {elapsed/len(structures):.1f} s")
print(f"\nResults saved to:")
print(f"  - screening_results.json")
print(f"  - screening_analysis.png")
print(f"  - optimized_structures/ (CIF files)")
print("=" * 60)

print("\n✓ Example 5 completed successfully!")
print("\n🎉 All eSEN examples completed!")
print("\nYou can now:")
print("  - Review results in screening_results.json")
print("  - Check optimized structures in optimized_structures/")
print("  - Adapt these scripts for your own materials")
