"""Phonon calculations and thermal properties"""

from typing import Optional, Union, List, Dict, Any
import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


def ase_to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    """Convert ASE Atoms to Phonopy Atoms."""
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.get_cell()
    )


def calculate_phonon(
    atoms: Atoms,
    calculator,
    supercell_matrix: Union[List[int], np.ndarray, int] = 2,
    displacement: float = 0.01,
    mesh: Optional[List[int]] = None,
    output_dir: Optional[str] = None
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
        
    Returns:
        Dictionary containing phonon object and basic properties
    """
    # Convert supercell_matrix to list
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
    
    # Calculate forces for displaced structures
    supercells = phonon.supercells_with_displacements
    forces_list = []
    
    for i, scell in enumerate(supercells):
        # Convert Phonopy atoms to ASE atoms
        ase_scell = Atoms(
            symbols=scell.get_chemical_symbols(),
            positions=scell.get_positions(),
            cell=scell.get_cell(),
            pbc=True
        )
        ase_scell.calc = calculator
        forces = ase_scell.get_forces()
        forces_list.append(forces)
    
    # Set forces to phonon
    phonon.forces = forces_list
    
    # Produce force constants
    phonon.produce_force_constants()
    
    # Calculate phonon DOS if mesh specified
    if mesh is None:
        mesh = [20, 20, 20]
    
    phonon.run_mesh(mesh)
    phonon.run_total_dos()
    
    # Save phonon files if output directory specified
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        phonon.save(f"{output_dir}/phonopy.yaml")
        phonon.write_yaml_force_constants(f"{output_dir}/force_constants.yaml")
    
    result = {
        "phonon": phonon,
        "supercell_matrix": supercell_matrix,
        "displacement": displacement,
        "mesh": mesh,
    }
    
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
        Dictionary with thermal properties
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
