"""
Phonon calculations and thermodynamic properties for eSEN models
"""

from ase import Atoms
from ase.calculators.calculator import Calculator
from typing import Dict, Any, Union, List, Optional
import numpy as np


class PhononTask:
    """
    Handler for phonon calculations using Phonopy.
    
    Computes:
    - Force constants
    - Phonon density of states
    - Thermodynamic properties (free energy, entropy, heat capacity)
    """
    
    def __init__(self, calculator: Calculator):
        """
        Initialize PhononTask.
        
        Args:
            calculator: ASE calculator (OCPCalculator for eSEN)
        """
        self.calculator = calculator
    
    def phonon(
        self,
        atoms: Atoms,
        supercell_matrix: Union[List[int], np.ndarray] = [2, 2, 2],
        mesh: Union[List[int], np.ndarray] = [20, 20, 20],
        displacement: float = 0.01,
        t_min: float = 0.0,
        t_max: float = 1000.0,
        t_step: float = 10.0
    ) -> Dict[str, Any]:
        """
        Calculate phonons and thermodynamic properties.
        
        Args:
            atoms: Primitive cell (should be well-optimized)
            supercell_matrix: Supercell size (3x3 matrix or [nx, ny, nz])
            mesh: k-point mesh for DOS calculation
            displacement: Atomic displacement amplitude (Å)
            t_min: Minimum temperature (K)
            t_max: Maximum temperature (K)
            t_step: Temperature step (K)
        
        Returns:
            Dictionary containing:
            - phonon: Phonopy object
            - force_constants: Force constants array
            - frequency_points: Frequency points (THz)
            - total_dos: Total phonon DOS
            - thermal: Dict with thermodynamic properties
              - temperatures: Temperature array (K)
              - free_energy: Helmholtz free energy (kJ/mol)
              - entropy: Entropy (J/(K·mol))
              - heat_capacity: Cv (J/(K·mol))
            - has_imaginary: Whether imaginary modes exist
            - imaginary_modes: Number of imaginary modes
        """
        try:
            from phonopy import Phonopy
            from phonopy.structure.atoms import PhonopyAtoms
        except ImportError:
            raise ImportError(
                "Phonopy is required for phonon calculations. "
                "Install with: pip install phonopy"
            )
        
        # Convert ASE Atoms to PhonopyAtoms
        phonopy_atoms = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            scaled_positions=atoms.get_scaled_positions()
        )
        
        # Create Phonopy object
        phonon = Phonopy(
            phonopy_atoms,
            supercell_matrix=supercell_matrix
        )
        
        # Generate displacements
        phonon.generate_displacements(distance=displacement)
        
        # Get supercells with displacements
        supercells = phonon.supercells_with_displacements
        
        # Calculate forces for each displaced supercell
        forces_list = []
        for supercell in supercells:
            # Convert PhonopyAtoms back to ASE Atoms
            ase_supercell = Atoms(
                symbols=supercell.symbols,
                cell=supercell.cell,
                scaled_positions=supercell.scaled_positions,
                pbc=True
            )
            ase_supercell.calc = self.calculator
            forces = ase_supercell.get_forces()
            forces_list.append(forces)
        
        # Set forces
        phonon.forces = forces_list
        
        # Produce force constants
        phonon.produce_force_constants()
        force_constants = phonon.force_constants
        
        # Calculate DOS
        phonon.run_mesh(mesh)
        phonon.run_total_dos()
        
        # Get DOS
        dos_dict = phonon.get_total_dos_dict()
        frequency_points = dos_dict['frequency_points']  # THz
        total_dos = dos_dict['total_dos']
        
        # Check for imaginary modes (negative frequencies)
        imaginary_modes = np.sum(frequency_points < -0.1)  # THz, threshold -0.1
        has_imaginary = imaginary_modes > 0
        
        # Calculate thermal properties
        phonon.run_thermal_properties(
            t_min=t_min,
            t_max=t_max,
            t_step=t_step
        )
        
        # Get thermal properties
        tp_dict = phonon.get_thermal_properties_dict()
        
        thermal = {
            'temperatures': tp_dict['temperatures'],  # K
            'free_energy': tp_dict['free_energy'],    # kJ/mol
            'entropy': tp_dict['entropy'],            # J/(K·mol)
            'heat_capacity': tp_dict['heat_capacity']  # J/(K·mol)
        }
        
        return {
            'phonon': phonon,
            'force_constants': force_constants,
            'frequency_points': frequency_points,
            'total_dos': total_dos,
            'thermal': thermal,
            'has_imaginary': has_imaginary,
            'imaginary_modes': int(imaginary_modes)
        }


def plot_phonon_dos(
    frequency_points: np.ndarray,
    total_dos: np.ndarray,
    output: str = 'phonon_dos.png',
    title: str = 'Phonon Density of States',
    xlim: Optional[tuple] = None,
    figsize: tuple = (8, 6)
):
    """
    Plot phonon density of states.
    
    Args:
        frequency_points: Frequency points (THz)
        total_dos: Total DOS
        output: Output file path
        title: Plot title
        xlim: x-axis limits (THz)
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(frequency_points, total_dos, 'b-', linewidth=1.5)
    ax.fill_between(frequency_points, 0, total_dos, alpha=0.3)
    
    ax.set_xlabel('Frequency (THz)', fontsize=12)
    ax.set_ylabel('DOS (states/THz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()


def plot_thermal_properties(
    temperatures: np.ndarray,
    heat_capacity: np.ndarray,
    output: str = 'heat_capacity.png',
    title: str = 'Heat Capacity',
    mass_per_formula: Optional[float] = None,
    figsize: tuple = (8, 6)
):
    """
    Plot heat capacity vs temperature.
    
    Args:
        temperatures: Temperature array (K)
        heat_capacity: Cv array (J/(K·mol))
        output: Output file path
        title: Plot title
        mass_per_formula: Molar mass (g/mol) for conversion to J/(K·g)
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if mass_per_formula is not None:
        # Convert to J/(K·g)
        cv_per_mass = heat_capacity / mass_per_formula
        ax.plot(temperatures, cv_per_mass, 'r-', linewidth=2)
        ax.set_ylabel('Cv (J/(K·g))', fontsize=12)
    else:
        ax.plot(temperatures, heat_capacity, 'r-', linewidth=2)
        ax.set_ylabel('Cv (J/(K·mol))', fontsize=12)
    
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
