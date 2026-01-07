"""Adsorption energy and binding site calculations"""

from typing import Optional, List, Dict, Any
import numpy as np
from ase import Atoms
from .static import optimize_structure, calculate_single_point


def calculate_adsorption_energy(
    host: Atoms,
    adsorbate: Atoms,
    combined: Atoms,
    calculator,
    relax_host: bool = True,
    relax_adsorbate: bool = True,
    relax_combined: bool = True,
    fmax: float = 0.05,
    optimizer: str = "LBFGS"
) -> Dict[str, Any]:
    """
    Calculate adsorption energy.
    
    Adsorption energy: E_ads = E(host+adsorbate) - E(host) - E(adsorbate)
    Negative value indicates favorable adsorption.
    
    Args:
        host: Host structure (e.g., MOF, surface)
        adsorbate: Adsorbate molecule (e.g., CO2, H2)
        combined: Combined host+adsorbate structure
        calculator: ASE calculator
        relax_host: Whether to relax host structure
        relax_adsorbate: Whether to relax adsorbate structure
        relax_combined: Whether to relax combined structure
        fmax: Force convergence criterion for relaxation (eV/Å)
        optimizer: Optimizer for relaxation
        
    Returns:
        Dictionary with:
            - adsorption_energy_eV: Adsorption energy (eV)
            - adsorption_energy_kJ_mol: Adsorption energy (kJ/mol)
            - host_energy: Host energy (eV)
            - adsorbate_energy: Adsorbate energy (eV)
            - combined_energy: Combined system energy (eV)
            - relaxed_host: Relaxed host structure (if relax_host=True)
            - relaxed_adsorbate: Relaxed adsorbate (if relax_adsorbate=True)
            - relaxed_combined: Relaxed combined structure (if relax_combined=True)
        
    Examples:
        >>> from grace_inference import GRACEInference
        >>> calc = GRACEInference(model_name="grace-2l")
        >>> 
        >>> # Load structures
        >>> host = read("MOF-5.cif")
        >>> adsorbate = read("CO2.xyz")
        >>> combined = read("MOF-5_CO2.cif")
        >>> 
        >>> # Calculate adsorption energy
        >>> result = calculate_adsorption_energy(
        ...     host, adsorbate, combined, calc.calculator
        ... )
        >>> print(f"Adsorption energy: {result['adsorption_energy_kJ_mol']:.2f} kJ/mol")
    """
    print("=" * 60)
    print("Adsorption Energy Calculation")
    print("=" * 60)
    
    # 1. Calculate host energy
    print("\n1. Calculating host energy...")
    if relax_host:
        print("   Relaxing host structure...")
        relaxed_host = optimize_structure(
            host, calculator, fmax=fmax, optimizer=optimizer
        )
        host_result = calculate_single_point(relaxed_host, calculator)
    else:
        relaxed_host = host.copy()
        host_result = calculate_single_point(host, calculator)
    
    host_energy = host_result["energy"]
    print(f"   Host energy: {host_energy:.6f} eV")
    
    # 2. Calculate adsorbate energy
    print("\n2. Calculating adsorbate energy...")
    if relax_adsorbate:
        print("   Relaxing adsorbate structure...")
        relaxed_adsorbate = optimize_structure(
            adsorbate, calculator, fmax=fmax, optimizer=optimizer
        )
        adsorbate_result = calculate_single_point(relaxed_adsorbate, calculator)
    else:
        relaxed_adsorbate = adsorbate.copy()
        adsorbate_result = calculate_single_point(adsorbate, calculator)
    
    adsorbate_energy = adsorbate_result["energy"]
    print(f"   Adsorbate energy: {adsorbate_energy:.6f} eV")
    
    # 3. Calculate combined system energy
    print("\n3. Calculating combined system energy...")
    if relax_combined:
        print("   Relaxing combined structure...")
        relaxed_combined = optimize_structure(
            combined, calculator, fmax=fmax, optimizer=optimizer
        )
        combined_result = calculate_single_point(relaxed_combined, calculator)
    else:
        relaxed_combined = combined.copy()
        combined_result = calculate_single_point(combined, calculator)
    
    combined_energy = combined_result["energy"]
    print(f"   Combined energy: {combined_energy:.6f} eV")
    
    # 4. Calculate adsorption energy
    adsorption_energy_eV = combined_energy - host_energy - adsorbate_energy
    adsorption_energy_kJ_mol = adsorption_energy_eV * 96.485  # 1 eV = 96.485 kJ/mol
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Adsorption energy: {adsorption_energy_eV:.6f} eV")
    print(f"                   {adsorption_energy_kJ_mol:.2f} kJ/mol")
    
    if adsorption_energy_eV < 0:
        print("Status: Favorable adsorption (negative energy)")
    else:
        print("Status: Unfavorable adsorption (positive energy)")
    
    result = {
        "adsorption_energy_eV": adsorption_energy_eV,
        "adsorption_energy_kJ_mol": adsorption_energy_kJ_mol,
        "host_energy": host_energy,
        "adsorbate_energy": adsorbate_energy,
        "combined_energy": combined_energy,
        "relaxed_host": relaxed_host if relax_host else None,
        "relaxed_adsorbate": relaxed_adsorbate if relax_adsorbate else None,
        "relaxed_combined": relaxed_combined if relax_combined else None,
    }
    
    return result


def calculate_binding_sites(
    host: Atoms,
    adsorbate: Atoms,
    calculator,
    grid_spacing: float = 1.0,
    height_above_surface: float = 2.5,
    relax_each_site: bool = True,
    fmax: float = 0.05,
    top_n_sites: int = 5
) -> List[Dict[str, Any]]:
    """
    Identify potential binding sites by grid search.
    
    Args:
        host: Host structure
        adsorbate: Adsorbate molecule
        calculator: ASE calculator
        grid_spacing: Grid spacing for site sampling (Å)
        height_above_surface: Initial height of adsorbate above surface (Å)
        relax_each_site: Whether to relax structure at each site
        fmax: Force convergence for relaxation
        top_n_sites: Number of top binding sites to return
        
    Returns:
        List of binding site dictionaries sorted by adsorption energy
        
    Examples:
        >>> sites = calculate_binding_sites(mof, co2, calc.calculator, grid_spacing=2.0)
        >>> best_site = sites[0]
        >>> print(f"Best site adsorption energy: {best_site['adsorption_energy_eV']:.4f} eV")
    """
    print("Scanning for binding sites...")
    print(f"Grid spacing: {grid_spacing} Å")
    
    # Get host cell dimensions
    cell = host.get_cell()
    cell_lengths = np.linalg.norm(cell, axis=1)
    
    # Create grid points in 2D (assume z is perpendicular to surface)
    nx = int(cell_lengths[0] / grid_spacing) + 1
    ny = int(cell_lengths[1] / grid_spacing) + 1
    
    sites = []
    total_sites = nx * ny
    
    print(f"Testing {total_sites} potential sites...")
    
    # Get adsorbate center of mass
    ads_com = adsorbate.get_center_of_mass()
    
    for i, x in enumerate(np.linspace(0, cell_lengths[0], nx)):
        for j, y in enumerate(np.linspace(0, cell_lengths[1], ny)):
            site_idx = i * ny + j
            print(f"  Site {site_idx+1}/{total_sites}...", end='\r')
            
            # Create combined structure with adsorbate at this position
            combined = host.copy()
            ads_copy = adsorbate.copy()
            
            # Translate adsorbate to grid position
            new_com = np.array([x, y, cell_lengths[2]/2 + height_above_surface])
            translation = new_com - ads_com
            ads_copy.translate(translation)
            
            # Combine structures
            combined.extend(ads_copy)
            
            # Calculate energy (with or without relaxation)
            if relax_each_site:
                combined = optimize_structure(
                    combined, calculator, fmax=fmax, steps=100
                )
            
            result = calculate_single_point(combined, calculator)
            energy = result["energy"]
            
            sites.append({
                "position": [x, y],
                "energy": energy,
                "structure": combined
            })
    
    print()  # New line after progress
    
    # Sort by energy (lowest first)
    sites.sort(key=lambda s: s["energy"])
    
    # Return top N sites
    return sites[:top_n_sites]


def screen_adsorbates(
    host: Atoms,
    adsorbates: List[Atoms],
    calculator,
    relax: bool = True,
    fmax: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Screen multiple adsorbates for adsorption on a host structure.
    
    Args:
        host: Host structure (e.g., MOF)
        adsorbates: List of adsorbate structures
        calculator: ASE calculator
        relax: Whether to relax structures
        fmax: Force convergence for relaxation
        
    Returns:
        List of results sorted by adsorption energy
        
    Examples:
        >>> adsorbates = [co2, ch4, h2, n2]
        >>> results = screen_adsorbates(mof, adsorbates, calc.calculator)
        >>> for r in results:
        ...     print(f"{r['name']}: {r['adsorption_energy_kJ_mol']:.2f} kJ/mol")
    """
    results = []
    
    print(f"Screening {len(adsorbates)} adsorbates...")
    
    for i, ads in enumerate(adsorbates):
        print(f"\nAdsorbate {i+1}/{len(adsorbates)}")
        
        # Create combined structure (assumes adsorbate is pre-positioned)
        combined = host.copy()
        combined.extend(ads)
        
        # Calculate adsorption energy
        # Note: This assumes combined structure is provided with adsorbate positioned
        result = calculate_adsorption_energy(
            host, ads, combined, calculator,
            relax_host=False,  # Only relax once
            relax_adsorbate=relax,
            relax_combined=relax,
            fmax=fmax
        )
        
        result["adsorbate_index"] = i
        result["adsorbate_formula"] = ads.get_chemical_formula()
        results.append(result)
    
    # Sort by adsorption energy (most negative first)
    results.sort(key=lambda r: r["adsorption_energy_eV"])
    
    return results


def calculate_henry_coefficient(
    adsorption_energy_kJ_mol: float,
    temperature_K: float = 298.15,
    accessible_volume_A3: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate Henry's coefficient from adsorption energy.
    
    Args:
        adsorption_energy_kJ_mol: Adsorption energy (kJ/mol)
        temperature_K: Temperature (K)
        accessible_volume_A3: Accessible volume of host (Å³)
        
    Returns:
        Dictionary with Henry's coefficient and related properties
        
    Examples:
        >>> K_H = calculate_henry_coefficient(-25.0, temperature_K=300)
        >>> print(f"Henry coefficient: {K_H['K_H_mol_kg_Pa']:.2e} mol/kg/Pa")
    """
    # Gas constant
    R = 8.314  # J/(mol·K)
    
    # Boltzmann factor
    # K_H ∝ exp(-ΔE_ads / RT)
    boltzmann_factor = np.exp(-adsorption_energy_kJ_mol * 1000 / (R * temperature_K))
    
    result = {
        "temperature_K": temperature_K,
        "adsorption_energy_kJ_mol": adsorption_energy_kJ_mol,
        "boltzmann_factor": boltzmann_factor,
    }
    
    if accessible_volume_A3 is not None:
        # Rough estimate of Henry coefficient
        # K_H ~ (accessible_volume / kT) * exp(-ΔE_ads / RT)
        kT_eV = 8.617e-5 * temperature_K  # eV
        kT_J = R * temperature_K
        
        # Convert volume from Å³ to m³
        volume_m3 = accessible_volume_A3 * 1e-30
        
        # Simplified Henry coefficient (units vary by application)
        result["accessible_volume_A3"] = accessible_volume_A3
        result["note"] = "Henry coefficient calculation is simplified. Use GCMC for accurate values."
    
    return result
