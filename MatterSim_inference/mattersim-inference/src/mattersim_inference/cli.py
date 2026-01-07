"""
MatterSim Inference - Command Line Interface

Provides the mattersim-infer command.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


def create_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(
        prog="mattersim-infer",
        description="MatterSim Inference - Material Property Inference Tool (MOFSimBench #3)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single point calculation
    sp_parser = subparsers.add_parser("single-point", help="Single point energy calculation")
    sp_parser.add_argument("structure", help="Structure file path")
    sp_parser.add_argument("--model", default="MatterSim-v1-5M", help="Model name")
    sp_parser.add_argument("--device", default="auto", help="Computing device")
    sp_parser.add_argument("--output", "-o", help="Output JSON file")
    
    # Structure optimization
    opt_parser = subparsers.add_parser("optimize", help="Structure optimization")
    opt_parser.add_argument("structure", help="Structure file path")
    opt_parser.add_argument("--model", default="MatterSim-v1-5M", help="Model name")
    opt_parser.add_argument("--device", default="auto", help="Computing device")
    opt_parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold")
    opt_parser.add_argument("--cell", action="store_true", help="Optimize cell")
    opt_parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps")
    opt_parser.add_argument("--output", "-o", help="Output structure file")
    
    # Molecular dynamics
    md_parser = subparsers.add_parser("md", help="Molecular dynamics simulation")
    md_parser.add_argument("structure", help="Structure file path")
    md_parser.add_argument("--model", default="MatterSim-v1-5M", help="Model name")
    md_parser.add_argument("--device", default="auto", help="Computing device")
    md_parser.add_argument("--ensemble", default="nvt", choices=["nve", "nvt", "npt"])
    md_parser.add_argument("--temp", type=float, default=300, help="Temperature (K)")
    md_parser.add_argument("--pressure", type=float, help="Pressure (GPa)")
    md_parser.add_argument("--steps", type=int, default=10000, help="Number of steps")
    md_parser.add_argument("--timestep", type=float, default=1.0, help="Time step (fs)")
    md_parser.add_argument("--trajectory", help="Trajectory file")
    md_parser.add_argument("--logfile", help="Log file")
    
    # Phonon calculation
    ph_parser = subparsers.add_parser("phonon", help="Phonon calculation")
    ph_parser.add_argument("structure", help="Primitive cell structure file")
    ph_parser.add_argument("--model", default="MatterSim-v1-5M", help="Model name")
    ph_parser.add_argument("--device", default="auto", help="Computing device")
    ph_parser.add_argument("--supercell", nargs=3, type=int, default=[2, 2, 2])
    ph_parser.add_argument("--mesh", nargs=3, type=int, default=[20, 20, 20])
    ph_parser.add_argument("--output", "-o", help="Output JSON file")
    
    # Bulk modulus
    bm_parser = subparsers.add_parser("bulk-modulus", help="Bulk modulus calculation")
    bm_parser.add_argument("structure", help="Structure file path")
    bm_parser.add_argument("--model", default="MatterSim-v1-5M", help="Model name")
    bm_parser.add_argument("--device", default="auto", help="Computing device")
    bm_parser.add_argument("--strain-range", type=float, default=0.05)
    bm_parser.add_argument("--npoints", type=int, default=11)
    bm_parser.add_argument("--output", "-o", help="Output JSON file")
    
    # Adsorption energy
    ads_parser = subparsers.add_parser("adsorption", help="Adsorption energy calculation")
    ads_parser.add_argument("structure", help="MOF structure file")
    ads_parser.add_argument("--model", default="MatterSim-v1-5M", help="Model name")
    ads_parser.add_argument("--device", default="auto", help="Computing device")
    ads_parser.add_argument("--gas", required=True, help="Gas molecule (CO2, H2O, ...)")
    ads_parser.add_argument("--site", nargs=3, type=float, required=True, help="Adsorption site")
    ads_parser.add_argument("--no-optimize", action="store_true", help="Do not optimize complex")
    ads_parser.add_argument("--fmax", type=float, default=0.05, help="Optimization threshold")
    ads_parser.add_argument("--output", "-o", help="Output JSON file")
    
    # Batch optimization
    batch_parser = subparsers.add_parser("batch-optimize", help="Batch structure optimization")
    batch_parser.add_argument("structures", nargs="+", help="List of structure files")
    batch_parser.add_argument("--model", default="MatterSim-v1-5M", help="Model name")
    batch_parser.add_argument("--device", default="auto", help="Computing device")
    batch_parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold")
    batch_parser.add_argument("--cell", action="store_true", help="Optimize cell")
    batch_parser.add_argument("--output-dir", "-o", default="optimized", help="Output directory")
    
    return parser


def cmd_single_point(args) -> int:
    """Single point calculation command."""
    from mattersim_inference import MatterSimInference
    
    print(f"MatterSim single point calculation: {args.structure}")
    calc = MatterSimInference(model_name=args.model, device=args.device)
    result = calc.single_point(args.structure)
    
    print(f"\n=== Calculation Results ===")
    print(f"Energy: {result['energy']:.6f} eV")
    print(f"Energy per atom: {result['energy_per_atom']:.6f} eV/atom")
    print(f"Max force: {result['max_force']:.6f} eV/Å")
    print(f"Pressure: {result['pressure']:.4f} GPa")
    
    if args.output:
        output_data = {
            "energy": result["energy"],
            "energy_per_atom": result["energy_per_atom"],
            "max_force": result["max_force"],
            "rms_force": result["rms_force"],
            "pressure": result["pressure"],
            "forces": result["forces"].tolist(),
            "stress": result["stress"].tolist(),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


def cmd_optimize(args) -> int:
    """Structure optimization command."""
    from mattersim_inference import MatterSimInference
    from ase.io import write
    
    print(f"MatterSim structure optimization: {args.structure}")
    calc = MatterSimInference(model_name=args.model, device=args.device)
    
    result = calc.optimize(
        args.structure,
        fmax=args.fmax,
        optimize_cell=args.cell,
        max_steps=args.max_steps
    )
    
    print(f"\n=== Optimization Results ===")
    print(f"Converged: {result['converged']}")
    print(f"Steps: {result['steps']}")
    print(f"Initial energy: {result['initial_energy']:.6f} eV")
    print(f"Final energy: {result['final_energy']:.6f} eV")
    print(f"Energy change: {result['energy_change']:.6f} eV")
    print(f"Final max force: {result['final_fmax']:.6f} eV/Å")
    
    if args.output:
        write(args.output, result["atoms"])
        print(f"\nOptimized structure saved to: {args.output}")
    
    return 0


def cmd_md(args) -> int:
    """Molecular dynamics command."""
    from mattersim_inference import MatterSimInference
    from ase.io import write
    
    print(f"MatterSim molecular dynamics: {args.structure}")
    print(f"Ensemble: {args.ensemble.upper()}, Temperature: {args.temp} K, Steps: {args.steps}")
    
    calc = MatterSimInference(model_name=args.model, device=args.device)
    
    final = calc.run_md(
        args.structure,
        ensemble=args.ensemble,
        temperature=args.temp,
        pressure=args.pressure,
        steps=args.steps,
        timestep=args.timestep,
        trajectory=args.trajectory,
        logfile=args.logfile
    )
    
    print(f"\n=== MD Completed ===")
    print(f"Final atom count: {len(final)}")
    
    return 0


def cmd_phonon(args) -> int:
    """Phonon calculation command."""
    from mattersim_inference import MatterSimInference
    
    print(f"MatterSim phonon calculation: {args.structure}")
    calc = MatterSimInference(model_name=args.model, device=args.device)
    
    result = calc.phonon(
        args.structure,
        supercell_matrix=args.supercell,
        mesh=args.mesh
    )
    
    print(f"\n=== Phonon Results ===")
    print(f"Imaginary frequencies: {result['has_imaginary']}")
    if result['has_imaginary']:
        print(f"Imaginary modes: {result['imaginary_modes']}")
    
    if args.output:
        output_data = {
            "has_imaginary": result["has_imaginary"],
            "imaginary_modes": result.get("imaginary_modes", 0),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


def cmd_bulk_modulus(args) -> int:
    """Bulk modulus command."""
    from mattersim_inference import MatterSimInference
    
    print(f"MatterSim bulk modulus calculation: {args.structure}")
    calc = MatterSimInference(model_name=args.model, device=args.device)
    
    result = calc.bulk_modulus(
        args.structure,
        strain_range=args.strain_range,
        npoints=args.npoints
    )
    
    print(f"\n=== Bulk Modulus Results ===")
    print(f"Bulk modulus: {result['bulk_modulus']:.2f} GPa")
    print(f"Equilibrium volume: {result['v0']:.2f} Å³")
    
    if args.output:
        output_data = {
            "bulk_modulus": result["bulk_modulus"],
            "v0": result["v0"],
            "e0": result["e0"],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


def cmd_adsorption(args) -> int:
    """Adsorption energy command."""
    from mattersim_inference import MatterSimInference
    
    print(f"MatterSim adsorption energy calculation (MOFSimBench #1): {args.structure}")
    print(f"Gas: {args.gas}, Site: {args.site}")
    
    calc = MatterSimInference(model_name=args.model, device=args.device)
    
    result = calc.adsorption_energy(
        args.structure,
        gas_molecule=args.gas,
        site_position=args.site,
        optimize_complex=not args.no_optimize,
        fmax=args.fmax
    )
    
    E_ads_kJ_mol = result["E_ads"] * 96.485
    
    print(f"\n=== Adsorption Energy Results ===")
    print(f"Adsorption energy: {result['E_ads']:.4f} eV ({E_ads_kJ_mol:.2f} kJ/mol)")
    print(f"MOF energy: {result['E_mof']:.4f} eV")
    print(f"Gas energy: {result['E_gas']:.4f} eV")
    print(f"Complex energy: {result['E_complex']:.4f} eV")
    
    if args.output:
        output_data = {
            "E_ads_eV": result["E_ads"],
            "E_ads_kJ_mol": E_ads_kJ_mol,
            "E_mof": result["E_mof"],
            "E_gas": result["E_gas"],
            "E_complex": result["E_complex"],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


def cmd_batch_optimize(args) -> int:
    """Batch optimization command."""
    from mattersim_inference import MatterSimInference
    from ase.io import write
    import os
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"MatterSim batch optimization: {len(args.structures)} structures")
    calc = MatterSimInference(model_name=args.model, device=args.device)
    
    results = []
    for struct_path in args.structures:
        name = Path(struct_path).stem
        print(f"\nOptimizing: {name}...")
        
        try:
            result = calc.optimize(
                struct_path,
                fmax=args.fmax,
                optimize_cell=args.cell
            )
            
            output_path = Path(args.output_dir) / f"{name}_opt.cif"
            write(str(output_path), result["atoms"])
            
            results.append({
                "name": name,
                "converged": result["converged"],
                "energy": result["final_energy"],
            })
            print(f"  -> Converged: {result['converged']}")
            
        except Exception as e:
            results.append({"name": name, "error": str(e)})
            print(f"  -> Error: {e}")
    
    # Save summary
    summary_path = Path(args.output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Batch Optimization Completed ===")
    print(f"Successful: {sum(1 for r in results if 'error' not in r)}/{len(results)}")
    print(f"Results saved to: {args.output_dir}/")
    
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point function."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        "single-point": cmd_single_point,
        "optimize": cmd_optimize,
        "md": cmd_md,
        "phonon": cmd_phonon,
        "bulk-modulus": cmd_bulk_modulus,
        "adsorption": cmd_adsorption,
        "batch-optimize": cmd_batch_optimize,
    }
    
    if args.command in commands:
        return commands[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
