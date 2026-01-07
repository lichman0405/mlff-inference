"""
Example 5: Bulk Modulus Calculation

This example demonstrates bulk modulus calculation via equation of state fitting.
"""

from ase.build import bulk
from equiformerv2_inference import EquiformerV2Inference

print("=" * 60)
print("EquiformerV2 Inference - Example 05: Bulk Modulus")
print("=" * 60)

# 1. Initialize model
print("\n1. Initializing EquiformerV2 model...")
calc = EquiformerV2Inference(
    model="equiformer_v2_31M",
    device="auto"
)

# 2. Create structure
print("\n2. Creating copper FCC structure...")
atoms = bulk("Cu", "fcc", a=3.6)
atoms = atoms * (2, 2, 2)  # 2x2x2 supercell

print(f"   Number of atoms: {len(atoms)}")
print(f"   Initial volume: {atoms.get_volume():.4f} Å³")

# 3. Optimize structure first
print("\n3. Optimizing structure...")
opt_result = calc.optimize(atoms, fmax=0.01, optimize_cell=True)
atoms = opt_result['atoms']
print(f"   Optimized volume: {atoms.get_volume():.4f} Å³")

# 4. Calculate bulk modulus
print("\n4. Calculating bulk modulus...")
print("   Strain range: ±5%")
print("   Number of points: 11")
print("   This may take a few minutes...")

result = calc.bulk_modulus(
    atoms,
    strain_range=0.05,
    npoints=11,
    eos="birchmurnaghan"
)

# 5. Display results
print("\n5. Calculation results:")
print(f"   Bulk modulus: {result['bulk_modulus']:.2f} GPa")
print(f"   Equilibrium volume: {result['v0']:.4f} Å³")
print(f"   Equilibrium energy: {result['e0']:.6f} eV")
print(f"   Equation of state: {result['eos']}")

# 6. Compare with experimental value
print("\n6. Comparison with experimental value:")
exp_bulk_modulus = 140  # Cu experimental value ~140 GPa
diff = abs(result['bulk_modulus'] - exp_bulk_modulus)
error_percent = diff / exp_bulk_modulus * 100
print(f"   Experimental value: ~{exp_bulk_modulus} GPa")
print(f"   Calculated value: {result['bulk_modulus']:.2f} GPa")
print(f"   Error: {error_percent:.1f}%")

# 7. Plot equation of state (if matplotlib available)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n7. Generating EOS plot...")
    
    volumes = result['volumes']
    energies = result['energies']
    v0 = result['v0']
    e0 = result['e0']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data points
    ax.plot(volumes, energies, 'bo', label='Calculated points', markersize=8)
    
    # Plot fitted curve
    from ase.eos import EquationOfState
    eos_fit = EquationOfState(volumes, energies, eos=result['eos'])
    eos_fit.fit()
    v_fit = np.linspace(volumes.min(), volumes.max(), 100)
    e_fit = [eos_fit.eos_function(v) for v in v_fit]
    ax.plot(v_fit, e_fit, 'r-', label='EOS fit', linewidth=2)
    
    # Mark equilibrium point
    ax.plot(v0, e0, 'rs', markersize=10, label=f'Equilibrium (V₀={v0:.2f} Å³)')
    
    ax.set_xlabel('Volume (Å³)', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title(f'Equation of State\nBulk Modulus = {result["bulk_modulus"]:.2f} GPa', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cu_eos.png', dpi=300)
    print("   ✓ Plot saved to cu_eos.png")
    
except ImportError:
    print("\n7. Matplotlib not available, skipping plot generation")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
