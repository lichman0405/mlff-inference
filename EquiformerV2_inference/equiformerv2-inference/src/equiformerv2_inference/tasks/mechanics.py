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
    eos: str = "birchmurnaghan"
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
    
    Returns:
        Dictionary with results:
            - bulk_modulus: Bulk modulus (GPa)
            - v0: Equilibrium volume (Å³)
            - e0: Equilibrium energy (eV)
            - B0: Bulk modulus derivative (dimensionless)
            - eos: EOS type used
            - volumes: Volume array (Å³)
            - energies: Energy array (eV)
            
    Examples:
        >>> result = calculate_bulk_modulus(atoms, calc, strain_range=0.05, npoints=11)
        >>> print(f"Bulk modulus: {result['bulk_modulus']:.2f} GPa")
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
        # For some EOS, this is a fitted parameter
        try:
            # Some EOS fits return B0
            B0 = eos_fit.eos_parameters.get('B0', None)
        except:
            B0 = None
        
        result = {
            "bulk_modulus": bulk_modulus_GPa,
            "v0": v0,
            "e0": e0,
            "B0": B0,
            "eos": eos,
            "volumes": volumes,
            "energies": energies,
        }
        
    except Exception as e:
        result = {
            "error": str(e),
            "volumes": volumes,
            "energies": energies,
        }
    
    return result


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
            - bulk_modulus: Bulk modulus from EOS fit (GPa)
            - note: Note about implementation
            
    Examples:
        >>> result = calculate_elastic_constants(atoms, calc, delta=0.01)
    """
    # Simplified implementation: only returns bulk modulus via EOS
    # Full elastic tensor calculation requires applying 6 independent strains
    # and calculating the stress-strain relationship
    
    bm_result = calculate_bulk_modulus(
        atoms, calculator, strain_range=0.03, npoints=7
    )
    
    return {
        "bulk_modulus": bm_result.get("bulk_modulus"),
        "note": "Full elastic tensor calculation not implemented. Use specialized tools for C_ij.",
    }


def calculate_shear_modulus(
    atoms: Atoms,
    calculator: Any,
    strain_range: float = 0.02,
    npoints: int = 7
) -> Dict[str, Any]:
    """
    Estimate shear modulus using simplified approach.
    
    Note: This provides a rough estimate. Accurate shear modulus requires
    full elastic constant tensor calculation.
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE Calculator
        strain_range: Shear strain range
        npoints: Number of strain points
        
    Returns:
        Dictionary with estimated shear modulus
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    
    original_cell = atoms.get_cell().copy()
    
    # Apply simple shear strain and measure stress
    strains = np.linspace(-strain_range, strain_range, npoints)
    stresses = []
    
    for strain in strains:
        # Create shear-strained structure (e.g., xy shear)
        shear_cell = original_cell.copy()
        shear_cell[0, 1] += strain * shear_cell[1, 1]  # xy shear
        
        strained_atoms = atoms.copy()
        strained_atoms.set_cell(shear_cell, scale_atoms=False)
        strained_atoms.calc = calculator
        
        # Get stress
        stress = strained_atoms.get_stress(voigt=True)
        stresses.append(stress[5])  # xy component
    
    stresses = np.array(stresses)
    
    # Fit linear stress-strain relationship: sigma = 2 * G * epsilon
    # G is shear modulus
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(strains, stresses)
    
    # Convert to GPa: slope is in eV/Å³, convert to GPa
    shear_modulus_GPa = abs(slope / 2.0) * 160.21766208
    
    return {
        "shear_modulus": shear_modulus_GPa,
        "r_squared": r_value**2,
        "note": "Simplified estimate from single shear deformation",
    }


def calculate_elastic_moduli(
    atoms: Atoms,
    calculator: Any,
    strain_range: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate elastic moduli (bulk and estimated shear).
    
    Args:
        atoms: ASE Atoms object
        calculator: ASE Calculator
        strain_range: Strain range for calculations
        
    Returns:
        Dictionary with bulk modulus and estimated elastic properties
        
    Examples:
        >>> moduli = calculate_elastic_moduli(atoms, calc)
        >>> print(f"Bulk modulus: {moduli['bulk_modulus']:.2f} GPa")
        >>> print(f"Young's modulus: {moduli['youngs_modulus']:.2f} GPa")
    """
    # Calculate bulk modulus
    bulk_result = calculate_bulk_modulus(atoms, calculator, strain_range=strain_range)
    B = bulk_result.get("bulk_modulus")
    
    if B is None:
        return {"error": "Failed to calculate bulk modulus"}
    
    # Estimate shear modulus (rough approximation)
    # For many materials: G ≈ 0.4 * B (very rough)
    G_estimate = 0.4 * B
    
    # Calculate Young's modulus: E = 9BG / (3B + G)
    E = 9 * B * G_estimate / (3 * B + G_estimate)
    
    # Calculate Poisson's ratio: nu = (3B - 2G) / (6B + 2G)
    nu = (3 * B - 2 * G_estimate) / (6 * B + 2 * G_estimate)
    
    return {
        "bulk_modulus": B,
        "shear_modulus_estimate": G_estimate,
        "youngs_modulus_estimate": E,
        "poisson_ratio_estimate": nu,
        "note": "Shear, Young's modulus and Poisson ratio are rough estimates",
    }


def plot_equation_of_state(
    bulk_modulus_result: Dict[str, Any],
    output_file: Optional[str] = None
) -> None:
    """
    Plot equation of state from bulk modulus calculation.
    
    Args:
        bulk_modulus_result: Result from calculate_bulk_modulus()
        output_file: Output file path for plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Cannot plot EOS.")
        return
    
    volumes = bulk_modulus_result["volumes"]
    energies = bulk_modulus_result["energies"]
    v0 = bulk_modulus_result["v0"]
    e0 = bulk_modulus_result["e0"]
    
    # Create fine grid for fitted curve
    v_fit = np.linspace(volumes.min(), volumes.max(), 100)
    
    # Fit EOS for plotting
    eos_type = bulk_modulus_result.get("eos", "birchmurnaghan")
    eos_fit = EquationOfState(volumes, energies, eos=eos_type)
    e_fit = eos_fit.fit()[0]  # Get fitted energies
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(volumes, energies, 'o', label='Calculated', markersize=8)
    plt.plot(v0, e0, 'r*', markersize=15, label=f'Minimum (V₀={v0:.2f} Ų)')
    
    plt.xlabel('Volume (ų)', fontsize=12)
    plt.ylabel('Energy (eV)', fontsize=12)
    plt.title(f'Equation of State ({eos_type})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"EOS plot saved to {output_file}")
    else:
        plt.show()
