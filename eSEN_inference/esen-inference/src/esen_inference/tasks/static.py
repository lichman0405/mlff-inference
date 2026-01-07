"""
Static calculations (single-point energy, forces, stress) for eSEN models
"""

from ase import Atoms
from ase.calculators.calculator import Calculator
import numpy as np
from typing import Dict, Any, List, Optional


class StaticTask:
    """
    Handler for single-point energy/force/stress calculations.
    
    This task computes energy, forces, and stress for a given structure
    without any relaxation or dynamics.
    """
    
    def __init__(self, calculator: Calculator):
        """
        Initialize StaticTask.
        
        Args:
            calculator: ASE calculator (OCPCalculator for eSEN)
        """
        self.calculator = calculator
    
    def single_point(
        self,
        atoms: Atoms,
        properties: List[str] = ['energy', 'forces', 'stress']
    ) -> Dict[str, Any]:
        """
        Perform single-point calculation.
        
        Args:
            atoms: Atoms object
            properties: List of properties to calculate
                       ['energy', 'forces', 'stress']
        
        Returns:
            Dictionary containing:
            - energy: Total energy (eV)
            - energy_per_atom: Energy per atom (eV/atom)
            - forces: Atomic forces (N_atoms, 3) in eV/Å
            - stress: Stress tensor in Voigt notation (6,) in eV/Å³
            - pressure: Pressure in GPa
            - max_force: Maximum force magnitude (eV/Å)
            - rms_force: RMS force (eV/Å)
        """
        # Attach calculator
        atoms.calc = self.calculator
        
        result = {}
        
        # Energy
        if 'energy' in properties:
            energy = atoms.get_potential_energy()
            result['energy'] = float(energy)
            result['energy_per_atom'] = float(energy / len(atoms))
        
        # Forces
        if 'forces' in properties:
            forces = atoms.get_forces()
            result['forces'] = forces
            result['max_force'] = float(np.max(np.linalg.norm(forces, axis=1)))
            result['rms_force'] = float(np.sqrt(np.mean(np.sum(forces**2, axis=1))))
        
        # Stress
        if 'stress' in properties and atoms.pbc.any():
            try:
                stress = atoms.get_stress(voigt=True)  # (6,) in eV/Å³
                result['stress'] = stress
                
                # Calculate pressure (negative of hydrostatic stress)
                # Pressure = -(σ_xx + σ_yy + σ_zz) / 3
                pressure_eV_A3 = -(stress[0] + stress[1] + stress[2]) / 3.0
                # Convert eV/Å³ to GPa: 1 eV/Å³ = 160.21766208 GPa
                pressure_GPa = pressure_eV_A3 * 160.21766208
                result['pressure'] = float(pressure_GPa)
                
                # Also provide virial if needed
                virial = -atoms.get_volume() * atoms.get_stress(voigt=False)  # (3, 3)
                result['virial'] = virial
                
            except Exception as e:
                # Some calculators may not support stress
                result['stress'] = None
                result['pressure'] = None
                result['virial'] = None
        
        return result
    
    def get_energy(self, atoms: Atoms) -> float:
        """
        Get total energy (convenience method).
        
        Args:
            atoms: Atoms object
        
        Returns:
            Total energy in eV
        """
        atoms.calc = self.calculator
        return atoms.get_potential_energy()
    
    def get_forces(self, atoms: Atoms) -> np.ndarray:
        """
        Get atomic forces (convenience method).
        
        Args:
            atoms: Atoms object
        
        Returns:
            Forces array (N_atoms, 3) in eV/Å
        """
        atoms.calc = self.calculator
        return atoms.get_forces()
    
    def get_stress(self, atoms: Atoms, voigt: bool = True) -> np.ndarray:
        """
        Get stress tensor (convenience method).
        
        Args:
            atoms: Atoms object
            voigt: If True, return Voigt notation (6,)
                   If False, return full tensor (3, 3)
        
        Returns:
            Stress in eV/Å³
        """
        atoms.calc = self.calculator
        return atoms.get_stress(voigt=voigt)
