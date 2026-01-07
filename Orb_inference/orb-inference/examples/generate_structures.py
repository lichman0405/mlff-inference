"""
Generate Example MOF Structures for Testing

This script creates simple example MOF structures for testing orb-inference.
"""

from ase import Atoms
from ase.build import bulk
from ase.io import write
from pathlib import Path
import numpy as np


def create_cu_btc_primitive():
    """
    Create a simplified Cu-BTC (HKUST-1) primitive cell.
    
    This is a highly simplified representation for testing purposes.
    Real Cu-BTC has formula Cu3(BTC)2 with large unit cell (~26 Å).
    """
    # Simplified cubic structure
    a = 26.34  # Approximate lattice constant (Å)
    
    # Cu paddlewheel positions (simplified)
    cu_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
    ]) * a
    
    # BTC linker C atoms (simplified)
    c_positions = np.array([
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
    ]) * a
    
    # O atoms connecting Cu to BTC (simplified)
    o_positions = np.array([
        [0.15, 0.15, 0.15],
        [0.35, 0.35, 0.15],
        [0.65, 0.65, 0.15],
        [0.85, 0.85, 0.15],
    ]) * a
    
    # Combine all positions
    positions = np.vstack([cu_positions, c_positions, o_positions])
    
    # Create atoms object
    symbols = ['Cu'] * len(cu_positions) + ['C'] * len(c_positions) + ['O'] * len(o_positions)
    
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[a, a, a],
        pbc=True
    )
    
    return atoms


def create_zif8_primitive():
    """
    Create a simplified ZIF-8 primitive cell.
    
    ZIF-8: Zn(MeIM)2 (MeIM = 2-methylimidazolate)
    Sodalite topology, cubic, a ≈ 17 Å
    """
    a = 16.99  # Lattice constant (Å)
    
    # Zn positions (tetrahedral nodes)
    zn_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]) * a
    
    # Imidazolate N positions (simplified)
    n_positions = np.array([
        [0.15, 0.15, 0.15],
        [0.35, 0.35, 0.15],
        [0.15, 0.35, 0.35],
        [0.35, 0.15, 0.35],
    ]) * a
    
    # Imidazolate C positions (simplified)
    c_positions = np.array([
        [0.20, 0.20, 0.20],
        [0.30, 0.30, 0.20],
        [0.20, 0.30, 0.30],
        [0.30, 0.20, 0.30],
    ]) * a
    
    positions = np.vstack([zn_positions, n_positions, c_positions])
    symbols = ['Zn'] * len(zn_positions) + ['N'] * len(n_positions) + ['C'] * len(c_positions)
    
    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[a, a, a],
        pbc=True
    )
    
    return atoms


def create_co2_molecule():
    """Create CO2 molecule."""
    positions = np.array([
        [0.0, 0.0, 0.0],    # C
        [-1.16, 0.0, 0.0],  # O
        [1.16, 0.0, 0.0],   # O
    ])
    
    atoms = Atoms(
        symbols=['C', 'O', 'O'],
        positions=positions,
        cell=[10, 10, 10],
        pbc=False
    )
    
    return atoms


def create_h2_molecule():
    """Create H2 molecule."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.74, 0.0, 0.0],  # H-H bond length ~0.74 Å
    ])
    
    atoms = Atoms(
        symbols=['H', 'H'],
        positions=positions,
        cell=[10, 10, 10],
        pbc=False
    )
    
    return atoms


def main():
    """Generate and save example structures."""
    
    # Create examples directory
    examples_dir = Path('examples/structures')
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating example structures...")
    
    # Cu-BTC
    print("\n1. Creating Cu-BTC (HKUST-1) simplified structure...")
    cu_btc = create_cu_btc_primitive()
    write(examples_dir / 'cu_btc_primitive.cif', cu_btc)
    print(f"   Saved: cu_btc_primitive.cif ({len(cu_btc)} atoms)")
    
    # ZIF-8
    print("\n2. Creating ZIF-8 simplified structure...")
    zif8 = create_zif8_primitive()
    write(examples_dir / 'zif8_primitive.cif', zif8)
    print(f"   Saved: zif8_primitive.cif ({len(zif8)} atoms)")
    
    # CO2 molecule
    print("\n3. Creating CO2 molecule...")
    co2 = create_co2_molecule()
    write(examples_dir / 'co2.xyz', co2)
    print(f"   Saved: co2.xyz ({len(co2)} atoms)")
    
    # H2 molecule
    print("\n4. Creating H2 molecule...")
    h2 = create_h2_molecule()
    write(examples_dir / 'h2.xyz', h2)
    print(f"   Saved: h2.xyz ({len(h2)} atoms)")
    
    # Simple test structures
    print("\n5. Creating simple test structures...")
    
    # Cu FCC
    cu = bulk('Cu', 'fcc', a=3.6)
    write(examples_dir / 'cu_fcc.cif', cu)
    print(f"   Saved: cu_fcc.cif ({len(cu)} atoms)")
    
    # Si diamond
    si = bulk('Si', 'diamond', a=5.43)
    write(examples_dir / 'si_diamond.cif', si)
    print(f"   Saved: si_diamond.cif ({len(si)} atoms)")
    
    print("\n" + "="*60)
    print("All example structures generated successfully!")
    print(f"Location: {examples_dir.absolute()}")
    print("="*60)
    
    # Create README
    readme_content = """# Example Structures

This directory contains simple example structures for testing orb-inference.

## MOF Structures

- **cu_btc_primitive.cif**: Simplified Cu-BTC (HKUST-1) primitive cell
  - Formula: Cu4C4O4 (simplified representation)
  - Lattice: Cubic, a ≈ 26.34 Å
  - Note: Highly simplified for testing, not chemically accurate

- **zif8_primitive.cif**: Simplified ZIF-8 primitive cell
  - Formula: Zn4N4C4 (simplified representation)
  - Lattice: Cubic, a ≈ 16.99 Å
  - Note: Simplified sodalite topology

## Molecules

- **co2.xyz**: CO2 molecule (linear, 3 atoms)
- **h2.xyz**: H2 molecule (2 atoms)

## Test Structures

- **cu_fcc.cif**: Cu face-centered cubic (1 atom)
- **si_diamond.cif**: Si diamond structure (2 atoms)

## Usage

```python
from orb_inference.utils.io import load_structure

# Load MOF
mof = load_structure('cu_btc_primitive.cif')

# Load molecule
co2 = load_structure('co2.xyz')
```

## Notes

- MOF structures are **simplified** for testing and demonstration
- Not suitable for production or publication-quality calculations
- For real MOF structures, use experimental CIF files from databases:
  - CoRE MOF Database
  - Cambridge Structural Database (CSD)
  - Materials Project
"""
    
    with open(examples_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"\nCreated README: {examples_dir / 'README.md'}")


if __name__ == '__main__':
    main()
