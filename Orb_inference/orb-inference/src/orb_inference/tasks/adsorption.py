"""Adsorption and guest molecule analysis."""

from typing import Optional, Union, List, Dict, Any, Tuple
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs


def calculate_adsorption_energy(
    host: Atoms,
    guest: Atoms,
    complex_atoms: Atoms,
    calculator,
    optimize_complex: bool = True,
    fmax: float = 0.05
) -> Dict[str, float]:
    """
    Calculate adsorption energy: E_ads = E(complex) - E(host) - E(guest)
    
    Args:
        host: MOF or host structure
        guest: Adsorbate molecule
        complex_atoms: Host + guest complex
        calculator: ORBCalculator instance
        optimize_complex: Whether to optimize complex structure
        fmax: Force convergence for optimization (eV/Å)
        
    Returns:
        Dictionary with:
            - E_complex: Energy of host+guest complex (eV)
            - E_host: Energy of isolated host (eV)
            - E_guest: Energy of isolated guest (eV)
            - E_ads: Adsorption energy (eV, negative = favorable)
            - E_ads_per_atom: Adsorption energy per guest atom (eV/atom)
            
    Note:
        Adsorption energy convention: E_ads < 0 means stable adsorption.
        For physisorption: typically -0.1 to -0.5 eV
        For chemisorption: typically -1 to -5 eV
        
    Examples:
        >>> result = calculate_adsorption_energy(
        ...     host=mof, 
        ...     guest=co2,
        ...     complex_atoms=mof_with_co2,
        ...     calculator=calc
        ... )
        >>> E_ads = result['E_ads']
        >>> print(f"Adsorption energy: {E_ads:.3f} eV")
    """
    from ase.optimize import LBFGS
    
    # Calculate complex energy
    complex_copy = complex_atoms.copy()
    complex_copy.calc = calculator
    
    if optimize_complex:
        print("Optimizing complex structure...")
        opt = LBFGS(complex_copy, logfile=None)
        opt.run(fmax=fmax)
    
    E_complex = complex_copy.get_potential_energy()
    print(f"  E(complex) = {E_complex:.6f} eV")
    
    # Calculate host energy
    host_copy = host.copy()
    host_copy.calc = calculator
    E_host = host_copy.get_potential_energy()
    print(f"  E(host) = {E_host:.6f} eV")
    
    # Calculate guest energy
    guest_copy = guest.copy()
    guest_copy.calc = calculator
    E_guest = guest_copy.get_potential_energy()
    print(f"  E(guest) = {E_guest:.6f} eV")
    
    # Calculate adsorption energy
    E_ads = E_complex - E_host - E_guest
    E_ads_per_atom = E_ads / len(guest)
    
    print(f"\nAdsorption Energy Results:")
    print(f"  E_ads = {E_ads:.6f} eV")
    print(f"  E_ads per guest atom = {E_ads_per_atom:.6f} eV/atom")
    
    if E_ads < 0:
        print(f"  → Stable adsorption (E_ads < 0)")
    else:
        print(f"  → Unstable adsorption (E_ads > 0)")
    
    return {
        "E_complex": E_complex,
        "E_host": E_host,
        "E_guest": E_guest,
        "E_ads": E_ads,
        "E_ads_per_atom": E_ads_per_atom,
    }


def analyze_coordination(
    atoms: Atoms,
    center_indices: Optional[List[int]] = None,
    neighbor_indices: Optional[List[int]] = None,
    cutoff_scale: float = 1.3,
    custom_cutoffs: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Analyze coordination environment around atoms.
    
    Uses ASE's NeighborList to identify coordinating atoms within
    cutoff distances (scaled natural cutoffs or custom values).
    
    Args:
        atoms: Structure to analyze
        center_indices: Indices of center atoms (e.g., metal sites)
                       If None, analyze all atoms
        neighbor_indices: Indices of potential neighbors
                         If None, consider all atoms
        cutoff_scale: Scaling factor for natural_cutoffs
        custom_cutoffs: Custom cutoff dict, e.g., {'Cu': 2.5, 'O': 1.8}
        
    Returns:
        Dictionary with:
            - coordination_numbers: Dict[int, int] mapping atom index to CN
            - neighbor_lists: Dict[int, List[int]] neighbors for each center
            - distances: Dict[int, List[float]] distances to neighbors (Å)
            
    Examples:
        >>> # Analyze coordination of metal sites (indices 0, 1)
        >>> result = analyze_coordination(
        ...     atoms, 
        ...     center_indices=[0, 1],
        ...     cutoff_scale=1.3
        ... )
        >>> cn = result['coordination_numbers']
        >>> print(f"Metal site 0: CN = {cn[0]}")
    """
    if center_indices is None:
        center_indices = list(range(len(atoms)))
    
    # Setup cutoffs
    if custom_cutoffs is not None:
        symbols = atoms.get_chemical_symbols()
        cutoffs = [custom_cutoffs.get(s, 2.0) for s in symbols]
    else:
        cutoffs = natural_cutoffs(atoms, mult=cutoff_scale)
    
    # Build neighbor list
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    coordination_numbers = {}
    neighbor_lists = {}
    distances = {}
    
    for center_idx in center_indices:
        # Get neighbors
        indices, offsets = nl.get_neighbors(center_idx)
        
        # Filter by neighbor_indices if specified
        if neighbor_indices is not None:
            mask = np.isin(indices, neighbor_indices)
            indices = indices[mask]
            offsets = offsets[mask]
        
        # Calculate distances
        center_pos = atoms.positions[center_idx]
        neighbor_positions = atoms.positions[indices] + offsets @ atoms.cell
        dists = np.linalg.norm(neighbor_positions - center_pos, axis=1)
        
        # Store results
        coordination_numbers[center_idx] = len(indices)
        neighbor_lists[center_idx] = indices.tolist()
        distances[center_idx] = dists.tolist()
    
    return {
        "coordination_numbers": coordination_numbers,
        "neighbor_lists": neighbor_lists,
        "distances": distances,
    }


def find_adsorption_sites(
    atoms: Atoms,
    guest_symbol: str = 'C',
    min_distance: float = 2.5,
    grid_spacing: float = 0.5
) -> List[np.ndarray]:
    """
    Find potential adsorption sites in MOF structure.
    
    Simple grid-based approach: places probe atoms on grid and 
    checks minimum distance to existing atoms.
    
    Args:
        atoms: MOF structure
        guest_symbol: Element symbol for distance check
        min_distance: Minimum distance to framework atoms (Å)
        grid_spacing: Grid spacing for probe placement (Å)
        
    Returns:
        List of potential adsorption site positions (Å)
        
    Note:
        This is a simplified approach. For production use, consider:
        - Zeo++ for pore analysis
        - RASPA for Monte Carlo sampling
        - Energy-based approaches with MLFF
        
    Examples:
        >>> sites = find_adsorption_sites(
        ...     mof, 
        ...     guest_symbol='C',
        ...     min_distance=2.5,
        ...     grid_spacing=0.5
        ... )
        >>> print(f"Found {len(sites)} potential sites")
    """
    from scipy.spatial.distance import cdist
    
    # Get cell dimensions
    cell = atoms.cell.array
    
    # Create grid
    nx = int(np.linalg.norm(cell[0]) / grid_spacing)
    ny = int(np.linalg.norm(cell[1]) / grid_spacing)
    nz = int(np.linalg.norm(cell[2]) / grid_spacing)
    
    grid_x = np.linspace(0, 1, nx, endpoint=False)
    grid_y = np.linspace(0, 1, ny, endpoint=False)
    grid_z = np.linspace(0, 1, nz, endpoint=False)
    
    # Fractional coordinates
    grid_frac = np.array(np.meshgrid(grid_x, grid_y, grid_z)).T.reshape(-1, 3)
    
    # Convert to cartesian
    grid_cart = grid_frac @ cell
    
    # Calculate distances to all atoms
    distances = cdist(grid_cart, atoms.positions)
    min_dists = distances.min(axis=1)
    
    # Filter by minimum distance
    valid_mask = min_dists >= min_distance
    valid_sites = grid_cart[valid_mask]
    
    print(f"Found {len(valid_sites)} potential adsorption sites")
    print(f"  Grid: {nx}×{ny}×{nz} = {len(grid_frac)} points")
    print(f"  Valid sites: {len(valid_sites)} (min_distance ≥ {min_distance} Å)")
    
    return valid_sites.tolist()


def calculate_binding_distance(
    complex_atoms: Atoms,
    host_indices: List[int],
    guest_indices: List[int]
) -> Dict[str, Any]:
    """
    Calculate binding distance between host and guest.
    
    Args:
        complex_atoms: Host + guest complex
        host_indices: Atom indices of binding site in host
        guest_indices: Atom indices of binding site in guest
        
    Returns:
        Dictionary with:
            - min_distance: Minimum distance between host and guest (Å)
            - average_distance: Average distance (Å)
            - closest_pair: (host_idx, guest_idx) of closest atoms
            
    Examples:
        >>> result = calculate_binding_distance(
        ...     complex_atoms,
        ...     host_indices=[0, 1, 2],  # Cu metal site
        ...     guest_indices=[100, 101]  # CO2 molecule
        ... )
        >>> d_min = result['min_distance']
    """
    from scipy.spatial.distance import cdist
    
    host_positions = complex_atoms.positions[host_indices]
    guest_positions = complex_atoms.positions[guest_indices]
    
    # Calculate distance matrix
    dist_matrix = cdist(host_positions, guest_positions)
    
    # Find minimum distance
    min_idx = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
    min_distance = dist_matrix[min_idx]
    
    # Closest pair
    closest_host_idx = host_indices[min_idx[0]]
    closest_guest_idx = guest_indices[min_idx[1]]
    
    # Average distance
    average_distance = dist_matrix.mean()
    
    return {
        "min_distance": min_distance,
        "average_distance": average_distance,
        "closest_pair": (closest_host_idx, closest_guest_idx),
        "distance_matrix": dist_matrix,
    }


def calculate_interaction_energy_components(
    complex_atoms: Atoms,
    guest_indices: List[int],
    calculator,
    distance_cutoff: float = 5.0
) -> Dict[str, Any]:
    """
    Decompose interaction energy by distance shells.
    
    This analyzes how much energy contribution comes from framework
    atoms at different distances from the guest molecule.
    
    Args:
        complex_atoms: Host + guest complex
        guest_indices: Indices of guest molecule atoms
        calculator: ORBCalculator instance
        distance_cutoff: Maximum distance for analysis (Å)
        
    Returns:
        Dictionary with distance-resolved energy contributions
        
    Note:
        This requires calculating energies with subsets of atoms,
        which may not be straightforward with GNN models.
        Implementation placeholder for future development.
        
    Examples:
        >>> result = calculate_interaction_energy_components(
        ...     complex_atoms, 
        ...     guest_indices=[100, 101, 102],
        ...     calculator=calc,
        ...     distance_cutoff=5.0
        ... )
    """
    raise NotImplementedError(
        "Energy decomposition requires model-specific approaches.\n"
        "For Orb models (GNS-based), global graph interactions make\n"
        "spatial decomposition challenging. Consider:\n"
        "  1. Attention weight analysis (if available)\n"
        "  2. Force-based interaction strength\n"
        "  3. Charge density analysis (requires extensions)\n"
    )
