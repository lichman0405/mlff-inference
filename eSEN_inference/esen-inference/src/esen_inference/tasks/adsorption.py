"""
Adsorption energy calculations for eSEN models
"""

from ase import Atoms
from ase.calculators.calculator import Calculator
from typing import Dict, Any, List, Optional
import numpy as np


class AdsorptionTask:
    """
    Handler for adsorption energy calculations.
    
    Calculates binding energy of guest molecules in MOFs:
    E_ads = E(host+guest) - E(host) - E(guest)
    """
    
    def __init__(self, calculator: Calculator):
        """
        Initialize AdsorptionTask.
        
        Args:
            calculator: ASE calculator (OCPCalculator for eSEN)
        """
        self.calculator = calculator
    
    def adsorption_energy(
        self,
        host: Atoms,
        guest: Atoms,
        complex_atoms: Atoms,
        optimize_complex: bool = True,
        optimize_host: bool = False,
        optimize_guest: bool = False,
        fmax: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate adsorption energy.
        
        Args:
            host: Host structure (MOF framework)
            guest: Guest molecule (adsorbate)
            complex_atoms: Host-guest complex
            optimize_complex: Whether to optimize complex
            optimize_host: Whether to optimize host
            optimize_guest: Whether to optimize guest
            fmax: Force convergence for optimization (eV/Å)
        
        Returns:
            Dictionary containing:
            - E_ads: Adsorption energy (eV, negative = stable)
            - E_ads_per_atom: Adsorption energy per guest atom (eV/atom)
            - E_complex: Complex energy (eV)
            - E_host: Host energy (eV)
            - E_guest: Guest energy (eV)
            - optimized_complex: Optimized complex structure
        """
        from esen_inference.tasks.optimization import OptimizationTask
        
        opt_task = OptimizationTask(self.calculator)
        
        # Optimize host if requested
        if optimize_host:
            host_opt = opt_task.optimize(
                host.copy(),
                fmax=fmax,
                relax_cell=False
            )
            E_host = host_opt['final_energy']
        else:
            host_calc = host.copy()
            host_calc.calc = self.calculator
            E_host = host_calc.get_potential_energy()
        
        # Optimize guest if requested
        if optimize_guest:
            guest_opt = opt_task.optimize(
                guest.copy(),
                fmax=fmax,
                relax_cell=False
            )
            E_guest = guest_opt['final_energy']
        else:
            guest_calc = guest.copy()
            guest_calc.calc = self.calculator
            E_guest = guest_calc.get_potential_energy()
        
        # Optimize complex if requested
        if optimize_complex:
            complex_opt = opt_task.optimize(
                complex_atoms.copy(),
                fmax=fmax,
                relax_cell=False
            )
            E_complex = complex_opt['final_energy']
            optimized_complex = complex_opt['atoms']
        else:
            complex_calc = complex_atoms.copy()
            complex_calc.calc = self.calculator
            E_complex = complex_calc.get_potential_energy()
            optimized_complex = complex_calc
        
        # Calculate adsorption energy
        # E_ads = E(complex) - E(host) - E(guest)
        # Negative value means stable adsorption
        E_ads = E_complex - E_host - E_guest
        E_ads_per_atom = E_ads / len(guest)
        
        return {
            'E_ads': float(E_ads),
            'E_ads_per_atom': float(E_ads_per_atom),
            'E_complex': float(E_complex),
            'E_host': float(E_host),
            'E_guest': float(E_guest),
            'optimized_complex': optimized_complex
        }


def find_adsorption_sites(
    atoms: Atoms,
    guest_symbol: str = 'He',
    min_distance: float = 2.5,
    grid_spacing: float = 0.5
) -> np.ndarray:
    """
    Find potential adsorption sites using grid-based approach.
    
    Args:
        atoms: MOF structure
        guest_symbol: Probe atom symbol
        min_distance: Minimum distance from framework atoms (Å)
        grid_spacing: Grid spacing for site search (Å)
    
    Returns:
        Array of positions (N_sites, 3) in Cartesian coordinates
    """
    from ase.geometry import get_distances
    
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    
    # Create grid
    nx = int(np.linalg.norm(cell[0]) / grid_spacing) + 1
    ny = int(np.linalg.norm(cell[1]) / grid_spacing) + 1
    nz = int(np.linalg.norm(cell[2]) / grid_spacing) + 1
    
    # Generate grid points
    grid_points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                frac_coords = np.array([i/nx, j/ny, k/nz])
                cart_coords = frac_coords @ cell
                grid_points.append(cart_coords)
    
    grid_points = np.array(grid_points)
    
    # Filter points based on distance from framework
    valid_sites = []
    for point in grid_points:
        # Calculate distances to all framework atoms
        distances = np.linalg.norm(positions - point, axis=1)
        
        # Keep if min distance is within acceptable range
        if np.min(distances) >= min_distance:
            valid_sites.append(point)
    
    return np.array(valid_sites)
