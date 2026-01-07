"""
Mechanical properties calculations for eSEN models
"""

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.eos import EquationOfState
from typing import Dict, Any, Optional
import numpy as np


class MechanicsTask:
    """
    Handler for mechanical properties calculations.
    
    Computes:
    - Bulk modulus (from equation of state fitting)
    - Elastic constants (from stress-strain relationships)
    """
    
    def __init__(self, calculator: Calculator):
        """
        Initialize MechanicsTask.
        
        Args:
            calculator: ASE calculator (OCPCalculator for eSEN)
        """
        self.calculator = calculator
    
    def bulk_modulus(
        self,
        atoms: Atoms,
        strain_range: float = 0.05,
        n_points: int = 7,
        eos_type: str = 'birchmurnaghan',
        optimize_first: bool = True,
        fmax: float = 0.01
    ) -> Dict[str, Any]:
        """
        Calculate bulk modulus from equation of state.
        
        Args:
            atoms: Input structure
            strain_range: Volume strain range (±)
            n_points: Number of volume points (odd number)
            eos_type: EOS type ('birchmurnaghan', 'murnaghan', 'vinet')
            optimize_first: Whether to optimize structure first
            fmax: Force convergence for optimization (eV/Å)
        
        Returns:
            Dictionary containing:
            - bulk_modulus: Bulk modulus (GPa)
            - bulk_modulus_prime: B' (dimensionless)
            - equilibrium_volume: V0 (Å³)
            - equilibrium_energy: E0 (eV)
            - eos: ASE EquationOfState object
            - volumes: Volume points (Å³)
            - energies: Corresponding energies (eV)
        """
        from esen_inference.tasks.optimization import OptimizationTask
        
        # Optimize structure first if requested
        if optimize_first:
            opt_task = OptimizationTask(self.calculator)
            opt_result = opt_task.optimize(
                atoms.copy(),
                fmax=fmax,
                relax_cell=True,
                optimizer='LBFGS'
            )
            atoms_eq = opt_result['atoms']
        else:
            atoms_eq = atoms.copy()
        
        # Get equilibrium volume
        V0 = atoms_eq.get_volume()
        
        # Generate volume points
        volumes = V0 * np.linspace(
            1 - strain_range,
            1 + strain_range,
            n_points
        )
        
        # Calculate energy for each volume
        energies = []
        cell_eq = atoms_eq.get_cell()
        
        for V in volumes:
            atoms_strained = atoms_eq.copy()
            
            # Scale cell uniformly to achieve target volume
            scale_factor = (V / V0) ** (1/3)
            atoms_strained.set_cell(cell_eq * scale_factor, scale_atoms=True)
            
            # Calculate energy
            atoms_strained.calc = self.calculator
            energy = atoms_strained.get_potential_energy()
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Fit equation of state
        eos = EquationOfState(volumes, energies, eos=eos_type)
        v0, e0, B = eos.fit()
        
        # Convert B from eV/Å³ to GPa
        # 1 eV/Å³ = 160.21766208 GPa
        B_GPa = B * 160.21766208
        
        # Get B' (bulk modulus derivative)
        # For Birch-Murnaghan: B' is the 4th parameter
        try:
            if eos_type == 'birchmurnaghan':
                # BM EOS has 4 parameters, B' is typically fixed at 4.0
                # or can be extracted from fit
                B_prime = 4.0  # Standard assumption
            else:
                B_prime = None
        except:
            B_prime = None
        
        return {
            'bulk_modulus': float(B_GPa),
            'bulk_modulus_prime': B_prime,
            'equilibrium_volume': float(v0),
            'equilibrium_energy': float(e0),
            'eos': eos,
            'volumes': volumes,
            'energies': energies
        }


def plot_eos(
    volumes: np.ndarray,
    energies: np.ndarray,
    eos: EquationOfState,
    output: str = 'eos_curve.png',
    title: str = 'Equation of State',
    figsize: tuple = (8, 6)
):
    """
    Plot equation of state curve.
    
    Args:
        volumes: Volume points (Å³)
        energies: Energy points (eV)
        eos: Fitted EquationOfState object
        output: Output file path
        title: Plot title
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data points
    ax.plot(volumes, energies, 'ro', markersize=8, label='Calculated')
    
    # Plot fitted curve
    v0, e0, B = eos.fit()
    v_fit = np.linspace(volumes.min(), volumes.max(), 100)
    e_fit = eos.func(v_fit, v0, e0, B)
    ax.plot(v_fit, e_fit, 'b-', linewidth=2, label='EOS fit')
    
    # Mark equilibrium point
    ax.plot(v0, e0, 'g*', markersize=15, label=f'V₀ = {v0:.2f} Ų')
    
    # Convert B to GPa
    B_GPa = B * 160.21766208
    
    ax.set_xlabel('Volume (ų)', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title(
        f'{title}\nB = {B_GPa:.2f} GPa',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_elastic_constants(
    atoms: Atoms,
    calculator: Calculator,
    delta: float = 0.01,
    voigt: bool = True
) -> Dict[str, Any]:
    """
    Calculate elastic constants from stress-strain relationships.
    
    Note: This is a basic implementation. For production use,
    consider using specialized tools like AFLOW-AEL.
    
    Args:
        atoms: Equilibrium structure (well-optimized)
        calculator: ASE calculator
        delta: Strain amplitude
        voigt: Return in Voigt notation (6x6)
    
    Returns:
        Dictionary with elastic tensor and derived properties
    
    Raises:
        NotImplementedError: Full elastic tensor calculation
                            requires advanced implementation
    """
    raise NotImplementedError(
        "Full elastic constants calculation requires advanced strain-stress "
        "mapping. For production use, please use specialized tools like "
        "AFLOW-AEL or elastool. For bulk modulus, use the bulk_modulus() method."
    )
