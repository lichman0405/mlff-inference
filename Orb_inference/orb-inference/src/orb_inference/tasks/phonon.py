"""Phonon calculations and thermal properties."""

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
        cell=atoms.cell.array,
        positions=atoms.positions,
        masses=atoms.get_masses()
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
        symbols=phonopy_atoms.symbols,
        cell=phonopy_atoms.cell,
        positions=phonopy_atoms.positions,
        pbc=True
    )


def calculate_phonon(
    atoms: Atoms,
    calculator,
    supercell_matrix: Union[List[int], List[List[int]]] = None,
    displacement: float = 0.01,
    mesh: List[int] = None,
    primitive_matrix: str = "auto",
) -> Dict[str, Any]:
    """
    Calculate phonon properties.
    
    Args:
        atoms: Primitive cell (should be optimized)
        calculator: ORBCalculator instance
        supercell_matrix: Supercell matrix, default [2, 2, 2]
        displacement: Displacement distance (Å)
        mesh: k-point mesh for DOS, default [20, 20, 20]
        primitive_matrix: Primitive matrix for Phonopy
        
    Returns:
        Dictionary with:
            - phonon: Phonopy object
            - frequency_points: Phonon frequencies (THz)
            - total_dos: Phonon DOS
            - supercell_matrix: Used supercell matrix
            
    Examples:
        >>> result = calculate_phonon(
        ...     atoms, calc,
        ...     supercell_matrix=[2, 2, 2],
        ...     mesh=[20, 20, 20]
        ... )
        >>> phonon = result['phonon']
    """
    if atoms.calc is None:
        atoms.calc = calculator
    
    # Default parameters
    if supercell_matrix is None:
        supercell_matrix = [2, 2, 2]
    if mesh is None:
        mesh = [20, 20, 20]
    
    # Convert to matrix if needed
    if isinstance(supercell_matrix[0], int):
        supercell_matrix = [[supercell_matrix[0], 0, 0],
                           [0, supercell_matrix[1], 0],
                           [0, 0, supercell_matrix[2]]]
    
    # Create Phonopy object
    phonon = Phonopy(
        ase_to_phonopy(atoms),
        supercell_matrix=supercell_matrix,
        primitive_matrix=primitive_matrix
    )
    
    # Generate displacements
    phonon.generate_displacements(distance=displacement)
    supercells = phonon.supercells_with_displacements
    
    print(f"Calculating forces for {len(supercells)} displaced supercells...")
    
    # Calculate forces for each displaced supercell
    forces = []
    for i, scell in enumerate(supercells):
        # Convert to ASE
        atoms_disp = phonopy_to_ase(scell)
        atoms_disp.calc = calculator
        
        # Calculate forces
        forces_disp = atoms_disp.get_forces()
        forces.append(forces_disp)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(supercells):
            print(f"  Progress: {i+1}/{len(supercells)}")
    
    # Set forces and produce force constants
    phonon.forces = forces
    phonon.produce_force_constants()
    
    # Calculate phonon DOS
    phonon.run_mesh(mesh=mesh)
    phonon.run_total_dos()
    dos_dict = phonon.get_total_dos_dict()
    
    return {
        "phonon": phonon,
        "frequency_points": dos_dict['frequency_points'],
        "total_dos": dos_dict['total_dos'],
        "supercell_matrix": supercell_matrix,
        "mesh": mesh,
    }


def calculate_thermal_properties(
    phonon: Phonopy,
    t_min: float = 0,
    t_max: float = 1000,
    t_step: float = 10
) -> Dict[str, np.ndarray]:
    """
    Calculate thermal properties from phonon.
    
    Args:
        phonon: Phonopy object (with force constants calculated)
        t_min: Minimum temperature (K)
        t_max: Maximum temperature (K)
        t_step: Temperature step (K)
        
    Returns:
        Dictionary with:
            - temperatures: Temperature points (K)
            - free_energy: Helmholtz free energy (kJ/mol)
            - entropy: Entropy (J/(K·mol))
            - heat_capacity: Heat capacity Cv (J/(K·mol))
            
    Examples:
        >>> tp = calculate_thermal_properties(phonon, t_min=0, t_max=1000, t_step=10)
        >>> temperatures = tp['temperatures']
        >>> Cv = tp['heat_capacity']
    """
    # Run thermal properties calculation
    phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)
    tp_dict = phonon.get_thermal_properties_dict()
    
    return {
        "temperatures": tp_dict['temperatures'],
        "free_energy": tp_dict['free_energy'],
        "entropy": tp_dict['entropy'],
        "heat_capacity": tp_dict['heat_capacity'],
    }


def plot_phonon_dos(
    frequency_points: np.ndarray,
    total_dos: np.ndarray,
    output: str = "phonon_dos.png"
):
    """
    Plot phonon density of states.
    
    Args:
        frequency_points: Frequency points (THz)
        total_dos: Phonon DOS
        output: Output figure path
        
    Examples:
        >>> plot_phonon_dos(result['frequency_points'], result['total_dos'])
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    plt.plot(frequency_points, total_dos, linewidth=2)
    plt.xlabel('Frequency (THz)', fontsize=12)
    plt.ylabel('Phonon DOS', fontsize=12)
    plt.title('Phonon Density of States', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    
    print(f"Phonon DOS plot saved to {output}")


def plot_thermal_properties(
    temperatures: np.ndarray,
    heat_capacity: np.ndarray,
    output: str = "thermal_properties.png",
    mass_per_formula: Optional[float] = None
):
    """
    Plot heat capacity vs temperature.
    
    Args:
        temperatures: Temperature points (K)
        heat_capacity: Heat capacity (J/(K·mol))
        output: Output figure path
        mass_per_formula: Molar mass (g/mol), for unit conversion to J/(K·g)
        
    Examples:
        >>> plot_thermal_properties(
        ...     tp['temperatures'], 
        ...     tp['heat_capacity'],
        ...     mass_per_formula=1000.0
        ... )
    """
    import matplotlib.pyplot as plt
    
    if mass_per_formula is not None:
        heat_capacity_per_mass = heat_capacity / mass_per_formula
        ylabel = 'Heat Capacity [J/(K·g)]'
    else:
        heat_capacity_per_mass = heat_capacity
        ylabel = 'Heat Capacity [J/(K·mol)]'
    
    plt.figure(figsize=(8, 6))
    plt.plot(temperatures, heat_capacity_per_mass, linewidth=2)
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title('Heat Capacity vs Temperature', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    
    print(f"Thermal properties plot saved to {output}")
