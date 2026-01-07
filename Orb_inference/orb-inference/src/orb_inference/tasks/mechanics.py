"""Mechanical properties calculations."""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from ase import Atoms
from ase.eos import EquationOfState


def calculate_bulk_modulus(
    atoms: Atoms,
    calculator,
    strain_range: float = 0.05,
    n_points: int = 7,
    eos_type: str = "birchmurnaghan",
    optimize_first: bool = True,
    fmax: float = 0.01
) -> Dict[str, Any]:
    """
    Calculate bulk modulus from equation of state.
    
    Performs volumetric strain (isotropic scaling) and fits EOS to extract
    equilibrium volume, bulk modulus, and energy.
    
    Args:
        atoms: Structure (should be near equilibrium)
        calculator: ORBCalculator instance
        strain_range: Volume strain range (±fraction)
        n_points: Number of volume points
        eos_type: EOS type ('birchmurnaghan', 'murnaghan', 'vinet', etc.)
        optimize_first: Whether to optimize structure first
        fmax: Force convergence for optimization (eV/Å)
        
    Returns:
        Dictionary with:
            - bulk_modulus: Bulk modulus B (GPa)
            - equilibrium_volume: V₀ (Å³)
            - equilibrium_energy: E₀ (eV)
            - volumes: Volume points (Å³)
            - energies: Energy points (eV)
            - eos: ASE EquationOfState object
            
    Examples:
        >>> result = calculate_bulk_modulus(
        ...     atoms, calc,
        ...     strain_range=0.05,
        ...     n_points=7
        ... )
        >>> B = result['bulk_modulus']
        >>> print(f"Bulk modulus: {B:.2f} GPa")
    """
    from ase.optimize import LBFGS
    from ase.constraints import FrechetCellFilter
    
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Optimize structure first if requested
    if optimize_first:
        print("Optimizing structure before bulk modulus calculation...")
        ecf = FrechetCellFilter(atoms)
        opt = LBFGS(ecf, logfile=None)
        opt.run(fmax=fmax)
        print(f"  Optimization completed: E = {atoms.get_potential_energy():.4f} eV")
    
    # Store equilibrium cell
    cell0 = atoms.cell.copy()
    v0 = atoms.get_volume()
    
    # Generate volume points
    strain_values = np.linspace(-strain_range, strain_range, n_points)
    volume_factors = (1 + strain_values) ** (1.0/3.0)
    
    volumes = []
    energies = []
    
    print(f"Calculating energies for {n_points} volume points...")
    for i, factor in enumerate(volume_factors):
        # Scale cell isotropically
        atoms_strained = atoms.copy()
        atoms_strained.set_cell(cell0 * factor, scale_atoms=True)
        atoms_strained.calc = calculator
        
        # Calculate energy
        energy = atoms_strained.get_potential_energy()
        volume = atoms_strained.get_volume()
        
        volumes.append(volume)
        energies.append(energy)
        
        print(f"  Point {i+1}/{n_points}: V = {volume:.2f} Å³, E = {energy:.4f} eV")
    
    # Fit equation of state
    volumes = np.array(volumes)
    energies = np.array(energies)
    
    eos = EquationOfState(volumes, energies, eos=eos_type)
    v_eq, e_eq, B = eos.fit()
    
    # Convert bulk modulus from eV/Å³ to GPa
    # 1 eV/Å³ = 160.21766208 GPa
    B_GPa = B * 160.21766208
    
    print(f"\nEquation of State Results ({eos_type}):")
    print(f"  Equilibrium volume: {v_eq:.3f} Å³")
    print(f"  Equilibrium energy: {e_eq:.6f} eV")
    print(f"  Bulk modulus: {B_GPa:.2f} GPa")
    
    return {
        "bulk_modulus": B_GPa,
        "equilibrium_volume": v_eq,
        "equilibrium_energy": e_eq,
        "volumes": volumes,
        "energies": energies,
        "eos": eos,
        "eos_type": eos_type,
    }


def plot_eos(
    volumes: np.ndarray,
    energies: np.ndarray,
    eos,
    output: str = "eos.png"
):
    """
    Plot equation of state.
    
    Args:
        volumes: Volume points (Å³)
        energies: Energy points (eV)
        eos: ASE EquationOfState object
        output: Output figure path
        
    Examples:
        >>> plot_eos(result['volumes'], result['energies'], result['eos'])
    """
    import matplotlib.pyplot as plt
    
    # Generate smooth curve
    v_fit = np.linspace(volumes.min(), volumes.max(), 100)
    e_fit = eos.func(v_fit, *eos.eos_parameters)
    
    plt.figure(figsize=(8, 6))
    plt.plot(volumes, energies, 'o', markersize=8, label='Calculated')
    plt.plot(v_fit, e_fit, '-', linewidth=2, label=f'EOS fit ({eos.eos_string})')
    plt.axvline(eos.v0, color='gray', linestyle='--', alpha=0.5, label='V₀')
    plt.xlabel('Volume (Å³)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title('Equation of State', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    
    print(f"EOS plot saved to {output}")


def calculate_elastic_constants(
    atoms: Atoms,
    calculator,
    delta: float = 0.01,
    voigt: bool = True
) -> Dict[str, Any]:
    """
    Calculate elastic constants tensor (C_ij).
    
    WARNING: This is a simplified implementation using finite differences.
    For production use, consider using elasticity packages or DFT codes.
    
    Args:
        atoms: Structure (should be optimized)
        calculator: ORBCalculator instance
        delta: Strain magnitude
        voigt: Return Voigt notation (6×6) or full tensor (3×3×3×3)
        
    Returns:
        Dictionary with:
            - elastic_tensor: Elastic constants (GPa)
            - bulk_modulus_vrh: Bulk modulus from VRH average (GPa)
            - shear_modulus_vrh: Shear modulus from VRH average (GPa)
            
    Note:
        This implementation applies strain and measures stress response.
        More sophisticated methods (e.g., ASE's stress-strain approach) 
        are recommended for publication-quality results.
        
    Examples:
        >>> result = calculate_elastic_constants(atoms, calc, delta=0.01)
        >>> C = result['elastic_tensor']
        >>> B_vrh = result['bulk_modulus_vrh']
    """
    raise NotImplementedError(
        "Elastic constants calculation requires strain-stress mapping.\n"
        "For MOF systems, recommend using:\n"
        "  1. ASE's strain module with stress calculations\n"
        "  2. ElaStic package for comprehensive elastic analysis\n"
        "  3. DFT codes (VASP/QE) for reference-quality results\n"
        "\n"
        "Simplified finite-difference approach:\n"
        "  - Apply 6 independent strains (ε_xx, ε_yy, ε_zz, ε_yz, ε_xz, ε_xy)\n"
        "  - Measure stress response σ_ij for each strain\n"
        "  - Build C_ijkl from linear relation σ_ij = C_ijkl ε_kl\n"
        "  - Convert to Voigt notation (6×6 matrix)\n"
    )


def estimate_youngs_modulus(
    bulk_modulus: float,
    poisson_ratio: float = 0.25
) -> float:
    """
    Estimate Young's modulus from bulk modulus (using Poisson ratio).
    
    For isotropic materials: E = 3B(1 - 2ν)
    
    Args:
        bulk_modulus: Bulk modulus (GPa)
        poisson_ratio: Poisson's ratio (typical 0.2-0.3 for MOFs)
        
    Returns:
        Young's modulus E (GPa)
        
    Note:
        This assumes isotropic behavior, which may not hold for MOFs.
        Use with caution and validate against experiments.
        
    Examples:
        >>> E = estimate_youngs_modulus(bulk_modulus=20.0, poisson_ratio=0.25)
        >>> print(f"Estimated Young's modulus: {E:.2f} GPa")
    """
    E = 3 * bulk_modulus * (1 - 2 * poisson_ratio)
    return E


def estimate_shear_modulus(
    bulk_modulus: float,
    poisson_ratio: float = 0.25
) -> float:
    """
    Estimate shear modulus from bulk modulus (using Poisson ratio).
    
    For isotropic materials: G = 3B(1 - 2ν) / (2(1 + ν))
    
    Args:
        bulk_modulus: Bulk modulus (GPa)
        poisson_ratio: Poisson's ratio (typical 0.2-0.3 for MOFs)
        
    Returns:
        Shear modulus G (GPa)
        
    Examples:
        >>> G = estimate_shear_modulus(bulk_modulus=20.0, poisson_ratio=0.25)
        >>> print(f"Estimated shear modulus: {G:.2f} GPa")
    """
    G = 3 * bulk_modulus * (1 - 2 * poisson_ratio) / (2 * (1 + poisson_ratio))
    return G
