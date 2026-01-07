"""
Adsorption energy calculation module.

Calculates adsorption energies of gas molecules in MOFs.
MatterSim performs **best** on this task (#1 in MOFSimBench).
"""

from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule
from ase.optimize import LBFGS


# Predefined gas molecules
GAS_MOLECULES = {
    "CO2": {
        "atoms": ["C", "O", "O"],
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.16],
            [0.0, 0.0, -1.16]
        ]
    },
    "H2O": {
        "atoms": ["O", "H", "H"],
        "positions": [
            [0.0, 0.0, 0.0],
            [0.757, 0.587, 0.0],
            [-0.757, 0.587, 0.0]
        ]
    },
    "CH4": {
        "atoms": ["C", "H", "H", "H", "H"],
        "positions": [
            [0.0, 0.0, 0.0],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629]
        ]
    },
    "N2": {
        "atoms": ["N", "N"],
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.098]
        ]
    },
    "H2": {
        "atoms": ["H", "H"],
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.74]
        ]
    },
    "CO": {
        "atoms": ["C", "O"],
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.128]
        ]
    },
    "NH3": {
        "atoms": ["N", "H", "H", "H"],
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, 0.94, 0.38],
            [0.81, -0.47, 0.38],
            [-0.81, -0.47, 0.38]
        ]
    },
}


def create_gas_molecule(gas: str) -> Atoms:
    """
    Create gas molecule.
    
    Args:
        gas: Gas molecule name
    
    Returns:
        Atoms: Gas molecule structure
    """
    gas = gas.upper()
    
    if gas in GAS_MOLECULES:
        mol_info = GAS_MOLECULES[gas]
        mol = Atoms(
            symbols=mol_info["atoms"],
            positions=mol_info["positions"]
        )
        mol.center(vacuum=10.0)
        mol.pbc = False
        return mol
    
    # Try using ASE built-in molecules
    try:
        mol = ase_molecule(gas)
        mol.center(vacuum=10.0)
        mol.pbc = False
        return mol
    except:
        raise ValueError(f"Unknown gas molecule: {gas}")


def calculate_adsorption_energy(
    mof: Atoms,
    gas: str,
    site: List[float],
    calculator: Any,
    optimize: bool = True,
    fmax: float = 0.05,
    max_steps: int = 200
) -> Dict[str, Any]:
    """
    Calculate adsorption energy.
    
    MatterSim performs **best** on this task (#1 in MOFSimBench).
    
    E_ads = E_complex - E_mof - E_gas
    
    Negative values indicate favorable adsorption (exothermic).
    
    Args:
        mof: MOF structure
        gas: Gas molecule name ("CO2", "H2O", "CH4", ...)
        site: Adsorption site coordinates [x, y, z]
        calculator: ASE Calculator
        optimize: Whether to optimize complex structure
        fmax: Force convergence threshold for optimization (eV/Å)
        max_steps: Maximum optimization steps
    
    Returns:
        dict: Adsorption energy results
            - E_ads: Adsorption energy (eV)
            - E_mof: MOF energy (eV)
            - E_gas: Gas molecule energy (eV)
            - E_complex: Complex energy (eV)
            - complex_atoms: Complex structure
    """
    # 1. Calculate MOF energy
    mof = mof.copy()
    mof.calc = calculator
    E_mof = mof.get_potential_energy()
    
    # 2. Create and calculate gas molecule energy
    gas_mol = create_gas_molecule(gas)
    
    # For molecules, use sufficiently large box
    gas_mol_calc = gas_mol.copy()
    gas_mol_calc.center(vacuum=15.0)
    gas_mol_calc.pbc = True
    gas_mol_calc.calc = calculator
    E_gas = gas_mol_calc.get_potential_energy()
    
    # 3. Create MOF + gas complex
    complex_atoms = mof.copy()
    
    # Move gas molecule to adsorption site
    gas_positions = gas_mol.get_positions()
    gas_center = gas_positions.mean(axis=0)
    shift = np.array(site) - gas_center
    gas_positions += shift
    
    # Add gas molecule
    for symbol, pos in zip(gas_mol.get_chemical_symbols(), gas_positions):
        complex_atoms.append(Atoms(symbol, positions=[pos]))
    
    complex_atoms.calc = calculator
    
    # 4. Optimize complex (optional)
    if optimize:
        opt = LBFGS(complex_atoms, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)
    
    E_complex = complex_atoms.get_potential_energy()
    
    # 5. Calculate adsorption energy
    E_ads = E_complex - E_mof - E_gas
    
    return {
        "E_ads": E_ads,
        "E_mof": E_mof,
        "E_gas": E_gas,
        "E_complex": E_complex,
        "complex_atoms": complex_atoms,
        "gas_molecule": gas,
        "site": site,
        "optimized": optimize,
    }


def scan_adsorption_sites(
    mof: Atoms,
    gas: str,
    sites: List[List[float]],
    calculator: Any,
    optimize: bool = True,
    fmax: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Scan multiple adsorption sites.
    
    Args:
        mof: MOF structure
        gas: Gas molecule name
        sites: List of adsorption sites
        calculator: ASE Calculator
        optimize: Whether to optimize
        fmax: Optimization threshold
    
    Returns:
        List[dict]: Adsorption energy results for each site
    """
    results = []
    
    for i, site in enumerate(sites):
        try:
            result = calculate_adsorption_energy(
                mof=mof,
                gas=gas,
                site=site,
                calculator=calculator,
                optimize=optimize,
                fmax=fmax
            )
            result["site_index"] = i
            results.append(result)
        except Exception as e:
            results.append({
                "site_index": i,
                "site": site,
                "error": str(e)
            })
    
    # Sort by adsorption energy
    valid_results = [r for r in results if "E_ads" in r]
    valid_results.sort(key=lambda x: x["E_ads"])
    
    return results
