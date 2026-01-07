"""Input/output utilities for structure files."""

from typing import Union, Optional, List
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read, write


def load_structure(file_path: Union[str, Path]) -> Atoms:
    """
    Load atomic structure from file.
    
    Supports common formats: CIF, POSCAR, XYZ, PDB, Trajectory, etc.
    
    Args:
        file_path: Path to structure file
        
    Returns:
        ASE Atoms object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
        
    Examples:
        >>> atoms = load_structure("MOF.cif")
        >>> print(len(atoms))
        156
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {file_path}")
    
    try:
        atoms = read(str(file_path))
        return atoms
    except Exception as e:
        raise ValueError(
            f"Failed to read structure from {file_path}. "
            f"Error: {e}"
        )


def save_structure(
    atoms: Atoms,
    output: Union[str, Path],
    format: Optional[str] = None
):
    """
    Save atomic structure to file.
    
    Args:
        atoms: ASE Atoms object to save
        output: Output file path
        format: File format (auto-detected from extension if None)
        
    Examples:
        >>> save_structure(atoms, "optimized.cif")
        >>> save_structure(atoms, "output.xyz", format="xyz")
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        write(str(output), atoms, format=format)
    except Exception as e:
        raise ValueError(
            f"Failed to write structure to {output}. "
            f"Error: {e}"
        )


def parse_structure_input(
    structure: Union[str, Path, Atoms]
) -> Atoms:
    """
    Parse structure input (file path or Atoms object).
    
    This provides a unified interface for functions that accept
    either file paths or Atoms objects.
    
    Args:
        structure: File path (str/Path) or ASE Atoms object
        
    Returns:
        ASE Atoms object
        
    Examples:
        >>> # From file
        >>> atoms = parse_structure_input("MOF.cif")
        
        >>> # From Atoms object
        >>> atoms = parse_structure_input(existing_atoms)
    """
    if isinstance(structure, Atoms):
        return structure.copy()
    elif isinstance(structure, (str, Path)):
        return load_structure(structure)
    else:
        raise TypeError(
            f"Structure must be file path (str/Path) or ASE Atoms object, "
            f"got {type(structure)}"
        )


def create_supercell(
    atoms: Atoms,
    supercell_matrix: Union[List[int], List[List[int]], np.ndarray]
) -> Atoms:
    """
    Create a supercell from primitive cell.
    
    Args:
        atoms: Primitive cell
        supercell_matrix: Supercell matrix
            - List of 3 integers: [n1, n2, n3] for diagonal matrix
            - 3x3 matrix: Full supercell transformation matrix
            
    Returns:
        Supercell Atoms object
        
    Examples:
        >>> # 2x2x2 supercell
        >>> supercell = create_supercell(atoms, [2, 2, 2])
        
        >>> # Custom supercell matrix
        >>> supercell = create_supercell(atoms, [[2,0,0], [0,2,0], [0,0,2]])
    """
    supercell_matrix = np.array(supercell_matrix)
    
    # Convert [n1, n2, n3] to diagonal matrix
    if supercell_matrix.ndim == 1:
        if len(supercell_matrix) != 3:
            raise ValueError(
                f"Supercell matrix must be [n1, n2, n3] or 3x3 matrix, "
                f"got shape {supercell_matrix.shape}"
            )
        supercell_matrix = np.diag(supercell_matrix)
    
    if supercell_matrix.shape != (3, 3):
        raise ValueError(
            f"Supercell matrix must be 3x3, got shape {supercell_matrix.shape}"
        )
    
    # Create supercell using ASE
    supercell = atoms.copy()
    supercell *= supercell_matrix.astype(int)
    
    return supercell


def atoms_to_dict(atoms: Atoms) -> dict:
    """
    Convert ASE Atoms object to dictionary (for JSON serialization).
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        Dictionary representation
        
    Examples:
        >>> atoms_dict = atoms_to_dict(atoms)
        >>> import json
        >>> json.dump(atoms_dict, open("structure.json", "w"))
    """
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
        "numbers": atoms.numbers.tolist(),
    }


def dict_to_atoms(atoms_dict: dict) -> Atoms:
    """
    Convert dictionary to ASE Atoms object.
    
    Args:
        atoms_dict: Dictionary with structure data
        
    Returns:
        ASE Atoms object
        
    Examples:
        >>> import json
        >>> atoms_dict = json.load(open("structure.json"))
        >>> atoms = dict_to_atoms(atoms_dict)
    """
    return Atoms(
        symbols=atoms_dict["symbols"],
        positions=atoms_dict["positions"],
        cell=atoms_dict["cell"],
        pbc=atoms_dict["pbc"],
    )


def get_formula(atoms: Atoms) -> str:
    """
    Get chemical formula from Atoms object.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        Chemical formula string
        
    Examples:
        >>> formula = get_formula(atoms)
        >>> print(formula)
        'Cu24O48C96H48'
    """
    return atoms.get_chemical_formula()


def get_volume(atoms: Atoms) -> float:
    """
    Get cell volume.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        Volume in Å³
        
    Examples:
        >>> volume = get_volume(atoms)
        >>> print(f"Volume: {volume:.2f} Å³")
    """
    return atoms.get_volume()
