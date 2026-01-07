"""Phonon calculations and thermal properties"""

from typing import Optional, Union, List, Dict, Any
import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


def ase_to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    """
    Convert ASE Atoms to Phonopy Atoms.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        PhonopyAtoms object
    """
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell()
    )


def phonopy_to_ase(phonopy_atoms: PhonopyAtoms) -> Atoms:
    """
    Convert Phonopy Atoms to ASE Atoms.
    
    Args:
        phonopy_atoms: PhonopyAtoms object
        
    Returns:
        ASE Atoms object
    """
    return Atoms(
        symbols=phonopy_atoms.get_chemical_symbols(),
        positions=phonopy_atoms.get_positions(),
        cell=phonopy_atoms.get_cell(),
        pbc=True
    )


def calculate_phonon(
    atoms: Atoms,
    calculator,
    supercell_matrix: Union[List[int], np.ndarray, int] = 2,
    displacement: float = 0.01,
    mesh: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    temperature_range: Optional[tuple] = None
) -> Dict[str, Any]:
    """
    Calculate phonon properties.
    
    Args:
        atoms: Input ASE Atoms object
        calculator: ASE calculator
        supercell_matrix: Supercell size (e.g., [2, 2, 2] or 2)
        displacement: Atomic displacement distance (Å)
        mesh: k-point mesh for phonon DOS (default: [20, 20, 20])
        output_dir: Output directory for phonon files
        temperature_range: Temperature range (t_min, t_max, t_step) in K
        
    Returns:
        Dictionary containing:
            - phonon: Phonopy object
            - supercell_matrix: Supercell matrix used
            - displacement: Displacement used
            - mesh: k-point mesh used
            - thermal_properties: Thermal properties (if temperature_range specified)
        
    Examples:
        >>> result = calculate_phonon(atoms, calc, supercell_matrix=2, mesh=[30, 30, 30])
        >>> phonon = result['phonon']
        >>> 
        >>> # With thermal properties
        >>> result = calculate_phonon(atoms, calc, supercell_matrix=2, 
        ...                          temperature_range=(0, 1000, 10))
    """
    # Convert supercell_matrix to proper format
    if isinstance(supercell_matrix, int):
        supercell_matrix = [[supercell_matrix, 0, 0],
                           [0, supercell_matrix, 0],
                           [0, 0, supercell_matrix]]
    elif isinstance(supercell_matrix, list) and len(supercell_matrix) == 3:
        if all(isinstance(x, int) for x in supercell_matrix):
            supercell_matrix = [[supercell_matrix[0], 0, 0],
                               [0, supercell_matrix[1], 0],
                               [0, 0, supercell_matrix[2]]]
    
    # Create Phonopy object
    phonopy_atoms = ase_to_phonopy(atoms)
    phonon = Phonopy(phonopy_atoms, supercell_matrix=supercell_matrix)
    
    # Generate displacements
    phonon.generate_displacements(distance=displacement)
    
    print(f"Calculating forces for {len(phonon.supercells_with_displacements)} displaced structures...")
    
    # Calculate forces for displaced structures
    supercells = phonon.supercells_with_displacements
    forces_list = []
    
    for i, scell in enumerate(supercells):
        print(f"  Structure {i+1}/{len(supercells)}...", end='\r')
        
        # Convert Phonopy atoms to ASE atoms
        ase_scell = phonopy_to_ase(scell)
        ase_scell.calc = calculator
        
        # Calculate forces
        forces = ase_scell.get_forces()
        forces_list.append(forces)
    
    print()  # New line after progress
    
    # Set forces to phonon
    phonon.forces = forces_list
    
    # Produce force constants
    print("Producing force constants...")
    phonon.produce_force_constants()
    
    # Calculate phonon DOS
    if mesh is None:
        mesh = [20, 20, 20]
    
    print(f"Running phonon calculation on {mesh} mesh...")
    phonon.run_mesh(mesh)
    phonon.run_total_dos()
    
    # Save phonon files if output directory specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving phonon data to {output_dir}...")
        phonon.save(f"{output_dir}/phonopy.yaml")
        phonon.write_yaml_force_constants(f"{output_dir}/force_constants.yaml")
    
    result = {
        "phonon": phonon,
        "supercell_matrix": supercell_matrix,
        "displacement": displacement,
        "mesh": mesh,
    }
    
    # Calculate thermal properties if requested
    if temperature_range is not None:
        t_min, t_max, t_step = temperature_range
        thermal = calculate_thermal_properties(phonon, t_min, t_max, t_step)
        result["thermal_properties"] = thermal
        
        if output_dir:
            save_thermal_properties(thermal, f"{output_dir}/thermal_properties.dat")
    
    return result


def calculate_thermal_properties(
    phonon: Phonopy,
    t_min: float = 0,
    t_max: float = 1000,
    t_step: float = 10
) -> Dict[str, np.ndarray]:
    """
    Calculate thermal properties from phonon object.
    
    Args:
        phonon: Phonopy object with force constants
        t_min: Minimum temperature (K)
        t_max: Maximum temperature (K)
        t_step: Temperature step (K)
        
    Returns:
        Dictionary with thermal properties:
            - temperatures: Temperature array (K)
            - free_energy: Helmholtz free energy (kJ/mol)
            - entropy: Entropy (J/(mol·K))
            - heat_capacity: Heat capacity at constant volume (J/(mol·K))
        
    Examples:
        >>> thermal = calculate_thermal_properties(phonon, t_min=0, t_max=500, t_step=5)
        >>> temps = thermal['temperatures']
        >>> cv = thermal['heat_capacity']
    """
    # Run thermal properties calculation
    phonon.run_thermal_properties(t_step=t_step, t_max=t_max, t_min=t_min)
    
    # Get thermal properties dictionary
    tp_dict = phonon.get_thermal_properties_dict()
    
    result = {
        "temperatures": tp_dict["temperatures"],  # K
        "free_energy": tp_dict["free_energy"],    # kJ/mol
        "entropy": tp_dict["entropy"],            # J/(mol·K)
        "heat_capacity": tp_dict["heat_capacity"], # J/(mol·K)
    }
    
    return result


def save_thermal_properties(
    thermal_properties: Dict[str, np.ndarray],
    filepath: str
) -> None:
    """
    Save thermal properties to file.
    
    Args:
        thermal_properties: Dictionary from calculate_thermal_properties
        filepath: Output file path
    """
    temps = thermal_properties["temperatures"]
    free_energy = thermal_properties["free_energy"]
    entropy = thermal_properties["entropy"]
    heat_capacity = thermal_properties["heat_capacity"]
    
    with open(filepath, 'w') as f:
        f.write("# Thermal Properties\n")
        f.write("# Temperature (K), Free Energy (kJ/mol), Entropy (J/mol/K), Heat Capacity (J/mol/K)\n")
        for i in range(len(temps)):
            f.write(f"{temps[i]:.2f}  {free_energy[i]:.6f}  {entropy[i]:.6f}  {heat_capacity[i]:.6f}\n")


def plot_phonon_bands(
    phonon: Phonopy,
    q_path: Optional[List] = None,
    labels: Optional[List[str]] = None,
    output_file: Optional[str] = None
) -> None:
    """
    Plot phonon band structure.
    
    Args:
        phonon: Phonopy object with force constants
        q_path: List of q-points for band structure
        labels: Labels for high-symmetry points
        output_file: Output file path for plot
        
    Examples:
        >>> plot_phonon_bands(phonon, output_file="phonon_bands.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Cannot plot phonon bands.")
        return
    
    if q_path is None:
        # Use automatic path
        phonon.auto_band_structure(plot=True, write_yaml=True, filename="band.yaml")
    else:
        # Manual path
        phonon.run_band_structure(q_path, labels=labels)
    
    # Plot
    phonon.plot_band_structure()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Phonon band structure saved to {output_file}")
    else:
        plt.show()


def check_negative_frequencies(
    phonon: Phonopy,
    threshold: float = -0.1
) -> Dict[str, Any]:
    """
    Check for negative (imaginary) phonon frequencies.
    
    Args:
        phonon: Phonopy object with calculated phonon frequencies
        threshold: Threshold for negative frequencies (THz)
        
    Returns:
        Dictionary with information about negative frequencies
    """
    # Get mesh frequencies
    mesh_dict = phonon.get_mesh_dict()
    frequencies = mesh_dict['frequencies']  # Shape: (nqpoints, nbands)
    
    # Find negative frequencies
    negative_mask = frequencies < threshold
    n_negative = np.sum(negative_mask)
    
    result = {
        "has_negative": n_negative > 0,
        "n_negative": int(n_negative),
        "min_frequency": float(np.min(frequencies)),
        "threshold": threshold,
    }
    
    if n_negative > 0:
        result["warning"] = (
            f"Found {n_negative} negative frequencies. "
            "This may indicate structural instability."
        )
    
    return result
