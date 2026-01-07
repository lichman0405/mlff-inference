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
        thermal_props = calculate_thermal_properties(
            phonon, t_min, t_max, t_step, output_dir
        )
        result["thermal_properties"] = thermal_props
    
    return result


def calculate_thermal_properties(
    phonon: Phonopy,
    t_min: float = 0,
    t_max: float = 1000,
    t_step: float = 10,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate thermal properties from phonon object.
    
    Args:
        phonon: Phonopy object
        t_min: Minimum temperature (K)
        t_max: Maximum temperature (K)
        t_step: Temperature step (K)
        output_dir: Output directory for thermal properties file
        
    Returns:
        Dictionary with thermal properties arrays
    """
    # Run thermal properties calculation
    phonon.run_thermal_properties(
        t_min=t_min,
        t_max=t_max,
        t_step=t_step
    )
    
    # Get thermal properties
    tp_dict = phonon.get_thermal_properties_dict()
    
    result = {
        "temperatures": tp_dict["temperatures"],  # K
        "free_energy": tp_dict["free_energy"],    # kJ/mol
        "entropy": tp_dict["entropy"],            # J/K/mol
        "heat_capacity": tp_dict["heat_capacity"] # J/K/mol
    }
    
    # Save thermal properties if output directory specified
    if output_dir:
        save_thermal_properties(result, output_dir)
    
    return result


def save_thermal_properties(
    thermal_props: Dict[str, Any],
    output_dir: str,
    filename: str = "thermal_properties.dat"
) -> None:
    """
    Save thermal properties to file.
    
    Args:
        thermal_props: Dictionary from calculate_thermal_properties
        output_dir: Output directory
        filename: Output filename
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write("# Temperature (K), Free Energy (kJ/mol), Entropy (J/K/mol), Heat Capacity (J/K/mol)\n")
        
        temps = thermal_props["temperatures"]
        fe = thermal_props["free_energy"]
        entropy = thermal_props["entropy"]
        cv = thermal_props["heat_capacity"]
        
        for i in range(len(temps)):
            f.write(f"{temps[i]:.2f}  {fe[i]:.6f}  {entropy[i]:.6f}  {cv[i]:.6f}\n")
    
    print(f"Thermal properties saved to {filepath}")


def plot_phonon_bands(
    phonon: Phonopy,
    path_labels: Optional[List[str]] = None,
    output_file: Optional[str] = None
) -> None:
    """
    Plot phonon band structure.
    
    Args:
        phonon: Phonopy object
        path_labels: High-symmetry path labels
        output_file: Output file path for plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Cannot plot phonon bands.")
        return
    
    # Get phonon band structure
    band_dict = phonon.get_band_structure_dict()
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for distances, frequencies, label in zip(
        band_dict["distances"],
        band_dict["frequencies"],
        band_dict.get("labels", [""] * len(band_dict["distances"]))
    ):
        ax.plot(distances, frequencies.T)
    
    ax.set_xlabel("Wave vector")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("Phonon Band Structure")
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Phonon band structure plot saved to {output_file}")
    else:
        plt.show()


def check_negative_frequencies(
    phonon: Phonopy,
    threshold: float = -0.1
) -> Dict[str, Any]:
    """
    Check for imaginary (negative) phonon frequencies.
    
    Args:
        phonon: Phonopy object
        threshold: Frequency threshold (THz), negative frequencies below this are reported
        
    Returns:
        Dictionary with:
            - has_negative: Boolean indicating if negative frequencies exist
            - min_frequency: Minimum frequency (THz)
            - negative_count: Number of negative frequency modes
    """
    # Get phonon DOS
    total_dos = phonon.get_total_dos_dict()
    frequencies = total_dos["frequency_points"]
    
    min_freq = np.min(frequencies)
    negative_freqs = frequencies[frequencies < threshold]
    
    result = {
        "has_negative": len(negative_freqs) > 0,
        "min_frequency": float(min_freq),
        "negative_count": len(negative_freqs),
    }
    
    if result["has_negative"]:
        print(f"WARNING: Found {result['negative_count']} negative frequency modes")
        print(f"Minimum frequency: {min_freq:.4f} THz")
        print("This may indicate structural instability.")
    else:
        print("No negative frequencies detected. Structure is dynamically stable.")
    
    return result
