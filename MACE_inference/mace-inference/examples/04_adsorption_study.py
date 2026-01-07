"""
Example 4: Gas Adsorption in MOFs

This example shows how to calculate adsorption energies.
"""

from ase.build import bulk, molecule
from mace_inference import MACEInference
import numpy as np

# Initialize calculator with D3 correction (important for van der Waals)
print("Initializing MACE calculator with D3 correction...")
calc = MACEInference(model="medium", device="auto", enable_d3=True)

# For this example, we'll use a simple cubic structure as a proxy for MOF
# In real applications, load actual MOF structures from CIF files
print("\nCreating model porous structure...")
# Create a simple porous structure (not a real MOF, just for demonstration)
mof = bulk('Cu', 'fcc', a=10.0)  # Large cell to simulate pore
mof = mof * (2, 2, 2)

print(f"MOF atoms: {len(mof)}")
print(f"MOF volume: {mof.get_volume():.2f} Å³")

# Test different gas molecules
gas_molecules = ["H2O", "CO2", "CH4", "N2"]

print("\n=== Calculating Adsorption Energies ===")

# Define adsorption site (center of the cell)
site_position = mof.get_center_of_mass()

results = {}

for gas in gas_molecules:
    print(f"\nCalculating {gas} adsorption...")
    
    try:
        result = calc.adsorption_energy(
            mof_structure=mof,
            gas_molecule=gas,
            site_position=site_position,
            optimize_complex=True,
            fmax=0.05
        )
        
        E_ads_eV = result['E_ads']
        E_ads_kJ_mol = E_ads_eV * 96.485  # Convert eV to kJ/mol
        
        results[gas] = {
            'E_ads_eV': E_ads_eV,
            'E_ads_kJ_mol': E_ads_kJ_mol
        }
        
        print(f"  E_ads = {E_ads_eV:.4f} eV ({E_ads_kJ_mol:.2f} kJ/mol)")
        
    except Exception as e:
        print(f"  Error: {e}")
        results[gas] = None

# Summary
print("\n=== Adsorption Energy Summary ===")
print(f"{'Gas':<10} {'E_ads (eV)':<15} {'E_ads (kJ/mol)':<20}")
print("-" * 45)

for gas, result in results.items():
    if result:
        print(f"{gas:<10} {result['E_ads_eV']:>10.4f}     {result['E_ads_kJ_mol']:>15.2f}")
    else:
        print(f"{gas:<10} {'Failed':<15}")

# Rank by binding strength (most negative = strongest binding)
if any(results.values()):
    valid_results = {k: v for k, v in results.items() if v is not None}
    ranked = sorted(valid_results.items(), key=lambda x: x[1]['E_ads_eV'])
    
    print("\n=== Ranking (strongest to weakest binding) ===")
    for i, (gas, data) in enumerate(ranked, 1):
        print(f"{i}. {gas}: {data['E_ads_eV']:.4f} eV")

print("\nNote: This is a demonstration with a simplified structure.")
print("For real MOF calculations, load actual MOF CIF files.")
