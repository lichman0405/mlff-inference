#!/usr/bin/env python
"""
Example 04: Adsorption Energy Calculation

Demonstrates how to use MatterSimInference to calculate adsorption energies.
MatterSim ranks #1 🥇 in adsorption energy calculations (MOFSimBench).
"""

from ase.build import bulk
from ase import Atoms
import numpy as np
from mattersim_inference import MatterSimInference


def create_simple_structure():
    """Create simple test structure."""
    # Create a simple cubic cell as an example
    atoms = bulk("Cu", "fcc", a=10.0, cubic=True)
    return atoms


def main():
    """Adsorption energy calculation example."""
    print("=" * 60)
    print("MatterSim Inference - Example 04: Adsorption Energy Calculation")
    print("MOFSimBench Adsorption Energy Ranking: #1 🥇")
    print("=" * 60)
    
    # 1. Initialize model
    print("\n1. Initializing MatterSim model...")
    calc = MatterSimInference(
        model_name="MatterSim-v1-5M",
        device="auto"
    )
    
    # 2. Create structure
    print("\n2. Creating test structure...")
    structure = create_simple_structure()
    print(f"   Number of atoms: {len(structure)}")
    print(f"   Cell volume: {structure.get_volume():.2f} Å³")
    
    # 3. Calculate CO2 adsorption energy
    print("\n3. Calculating CO2 adsorption energy...")
    site_position = [5.0, 5.0, 5.0]  # Cell center
    
    result = calc.adsorption_energy(
        mof_structure=structure,
        gas_molecule="CO2",
        site_position=site_position,
        optimize_complex=True,
        fmax=0.05
    )
    
    # 4. Display results
    E_ads_kJ_mol = result['E_ads'] * 96.485
    
    print("\n4. Calculation results:")
    print(f"   Adsorption energy: {result['E_ads']:.4f} eV ({E_ads_kJ_mol:.2f} kJ/mol)")
    print(f"   Structure energy: {result['E_mof']:.4f} eV")
    print(f"   CO2 energy: {result['E_gas']:.4f} eV")
    print(f"   Complex energy: {result['E_complex']:.4f} eV")
    
    if E_ads_kJ_mol < 0:
        print("\n   ✓ Negative adsorption energy indicates exothermic adsorption, thermodynamically favorable")
    else:
        print("\n   ✗ Positive adsorption energy indicates endothermic adsorption")
    
    # 5. Test other gases
    print("\n5. Testing other gas molecules...")
    gases = ["H2O", "CH4", "N2"]
    
    for gas in gases:
        try:
            result = calc.adsorption_energy(
                mof_structure=structure,
                gas_molecule=gas,
                site_position=site_position,
                optimize_complex=False  # Don't optimize to speed up
            )
            E_ads_kJ = result['E_ads'] * 96.485
            print(f"   {gas}: {result['E_ads']:.4f} eV ({E_ads_kJ:.2f} kJ/mol)")
        except Exception as e:
            print(f"   {gas}: Calculation failed - {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
