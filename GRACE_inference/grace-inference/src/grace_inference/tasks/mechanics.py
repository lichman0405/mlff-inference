"""Mechanical properties calculations"""

from typing import Any, Dict, Optional
import numpy as np
from ase import Atoms
from ase.eos import EquationOfState


def calculate_bulk_modulus(
    atoms: Atoms,
    calculator: Any,
    strain_range: float = 0.05,
    npoints: int = 11,
    eos: str = "birchmurnaghan",
    plot: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate bulk modulus from equation of state.
    
    Fits equation of state from energies at different volumes to obtain bulk modulus.
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE Calculator
        strain_range: Strain range (±), e.g., 0.05 means ±5%
        npoints: Number of volume sampling points
        eos: Equation of state type
            - "birchmurnaghan": Birch-Murnaghan (default, most common)
            - "vinet": Vinet EOS
            - "murnaghan": Murnaghan EOS
            - "sjeos": Stabilized Jellium EOS
            - "taylor": Taylor expansion
        plot: Whether to generate plot
        output_dir: Output directory for results
    
    Returns:
        Dictionary with results:
            - bulk_modulus_GPa: Bulk modulus (GPa)
            - v0: Equilibrium volume (Å³)
            - e0: Equilibrium energy (eV)
            - B0: Bulk modulus derivative (dimensionless)
            - eos: EOS type used
            - volumes: Volume array (Å³)
            - energies: Energy array (eV)
            
    Examples:
        >>> result = calculate_bulk_modulus(atoms, calc, strain_range=0.05, npoints=11)
        >>> print(f"Bulk modulus: {result['bulk_modulus_GPa']:.2f} GPa")
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    
    # Original volume and cell
    original_volume = atoms.get_volume()
    original_cell = atoms.get_cell().copy()
    
    # Generate different volumes by scaling the cell
    strains = np.linspace(-strain_range, strain_range, npoints)
    volumes = []
    energies = []
    
    print(f"Calculating {npoints} energy-volume points...")
    for i, strain in enumerate(strains):
        # Isotropic scaling: V' = V * (1 + strain)
        # Cell scaling factor: (1 + strain)^(1/3)
        scale = (1 + strain) ** (1/3)
        scaled_cell = original_cell * scale
        
        # Create scaled structure
        scaled_atoms = atoms.copy()
        scaled_atoms.set_cell(scaled_cell, scale_atoms=True)
        scaled_atoms.calc = calculator
        
        # Calculate energy
        energy = scaled_atoms.get_potential_energy()
        volume = scaled_atoms.get_volume()
        
        volumes.append(volume)
        energies.append(energy)
        
        print(f"  Point {i+1}/{npoints}: V={volume:.3f} Å³, E={energy:.6f} eV", end='\r')
    
    print()  # New line after progress
    
    volumes = np.array(volumes)
    energies = np.array(energies)
    
    # Fit equation of state
    try:
        eos_fit = EquationOfState(volumes, energies, eos=eos)
        v0, e0, B = eos_fit.fit()
        
        # Convert bulk modulus: eV/Å³ -> GPa
        # 1 eV/Å³ = 160.21766208 GPa
        bulk_modulus_GPa = B * 160.21766208
        
        # Get B0 (pressure derivative of bulk modulus) if available
        try:
            B0 = eos_fit.eos_parameters.get('B0', None)
        except:
            B0 = None
        
        result = {
            "bulk_modulus_GPa": bulk_modulus_GPa,
            "v0": v0,
            "e0": e0,
            "B0": B0,
            "eos": eos,
            "volumes": volumes,
            "energies": energies,
        }
        
        print(f"\nBulk modulus: {bulk_modulus_GPa:.2f} GPa")
        print(f"Equilibrium volume: {v0:.3f} Å³")
        print(f"Equilibrium energy: {e0:.6f} eV")
        
        # Plot if requested
        if plot:
            plot_equation_of_state(result, output_dir)
        
    except Exception as e:
        result = {
            "error": str(e),
            "volumes": volumes,
            "energies": energies,
        }
        print(f"Error fitting EOS: {e}")
    
    return result


def plot_equation_of_state(
    eos_result: Dict[str, Any],
    output_dir: Optional[str] = None
) -> None:
    """
    Plot equation of state fitting results.
    
    Args:
        eos_result: Result dictionary from calculate_bulk_modulus
        output_dir: Output directory for plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Cannot plot EOS.")
        return
    
    if "error" in eos_result:
        print(f"Cannot plot: {eos_result['error']}")
        return
    
    volumes = eos_result["volumes"]
    energies = eos_result["energies"]
    v0 = eos_result["v0"]
    e0 = eos_result["e0"]
    
    # Create fine grid for fitted curve
    v_fine = np.linspace(volumes.min(), volumes.max(), 100)
    
    # Reconstruct EOS fit
    from ase.eos import EquationOfState
    eos_fit = EquationOfState(volumes, energies, eos=eos_result["eos"])
    e_fine = eos_fit.func(v_fine)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(volumes, energies, 'o', label='Calculated', markersize=8)
    ax.plot(v_fine, e_fine, '-', label=f'{eos_result["eos"]} fit')
    ax.plot(v0, e0, 'r*', markersize=15, label='Equilibrium')
    
    ax.set_xlabel('Volume (Å³)')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(f'Equation of State\nB = {eos_result["bulk_modulus_GPa"]:.2f} GPa')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "eos.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"EOS plot saved to {output_file}")
    else:
        plt.show()


def calculate_elastic_constants(
    atoms: Atoms,
    calculator: Any,
    delta: float = 0.01,
    voigt: bool = True
) -> Dict[str, Any]:
    """
    Calculate elastic constant tensor.
    
    Note: This is a simplified implementation. For accurate elastic constants,
    consider using specialized tools like elastool or AELAS.
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE Calculator
        delta: Strain increment for numerical differentiation
        voigt: If True, return Voigt-averaged moduli
    
    Returns:
        Dictionary with elastic properties:
            - bulk_modulus_GPa: Bulk modulus from EOS fit (GPa)
            - note: Note about implementation
            
    Examples:
        >>> result = calculate_elastic_constants(atoms, calc, delta=0.01)
    """
    # Simplified implementation: only returns bulk modulus via EOS
    print("Note: Full elastic tensor calculation not implemented.")
    print("Calculating bulk modulus via equation of state instead.")
    
    result = calculate_bulk_modulus(atoms, calculator)
    result["note"] = "Full elastic tensor not implemented. Use specialized tools for C_ij."
    
    return result


def calculate_shear_modulus(
    atoms: Atoms,
    calculator: Any
) -> Dict[str, Any]:
    """
    Calculate shear modulus (simplified).
    
    Note: Accurate shear modulus calculation requires full elastic tensor.
    This is a placeholder for future implementation.
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE Calculator
    
    Returns:
        Dictionary with note about implementation
    """
    return {
        "note": "Shear modulus calculation requires full elastic tensor.",
        "recommendation": "Use specialized tools like elastool or AELAS."
    }


def calculate_elastic_moduli(
    atoms: Atoms,
    calculator: Any
) -> Dict[str, Any]:
    """
    Calculate elastic moduli (Young's modulus, Poisson's ratio, etc.).
    
    Note: These require full elastic tensor. This is a placeholder.
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE Calculator
    
    Returns:
        Dictionary with available results
    """
    result = calculate_bulk_modulus(atoms, calculator)
    result["note"] = "Full elastic moduli require elastic tensor calculation."
    
    return result
