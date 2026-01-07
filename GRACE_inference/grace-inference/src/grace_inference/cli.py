"""Command-line interface for GRACE inference"""

import argparse
import sys
from pathlib import Path


def create_parser():
    """Create argument parser for GRACE CLI."""
    parser = argparse.ArgumentParser(
        description="GRACE Inference - High-level CLI for GRACE force field calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-point calculation
  grace-inference single-point structure.cif --model grace-2l --device cuda
  
  # Structure optimization
  grace-inference optimize structure.cif --fmax 0.01 --cell --output optimized.cif
  
  # Molecular dynamics
  grace-inference md structure.cif --ensemble nvt --temp 300 --steps 10000 --trajectory md.traj
  
  # Phonon calculation
  grace-inference phonon structure.cif --supercell 3 3 3 --mesh 30 30 30
  
  # Bulk modulus
  grace-inference bulk-modulus structure.cif --points 15 --eos birchmurnaghan
  
  # Adsorption energy
  grace-inference adsorption MOF-5.cif CO2.xyz MOF-5_CO2.cif --relax-all
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ==================== Single-Point Calculation ====================
    sp_parser = subparsers.add_parser(
        'single-point',
        help='Calculate single-point energy, forces, and stress'
    )
    sp_parser.add_argument('structure', type=str, help='Structure file path')
    sp_parser.add_argument('--model', default='grace-2l', help='GRACE model name or path')
    sp_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                          help='Computing device')
    sp_parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    
    # ==================== Structure Optimization ====================
    opt_parser = subparsers.add_parser(
        'optimize',
        help='Optimize atomic structure'
    )
    opt_parser.add_argument('structure', type=str, help='Structure file path')
    opt_parser.add_argument('--model', default='grace-2l', help='GRACE model name')
    opt_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                           help='Computing device')
    opt_parser.add_argument('--fmax', default=0.05, type=float,
                           help='Force convergence criterion (eV/Å)')
    opt_parser.add_argument('--steps', default=500, type=int,
                           help='Maximum optimization steps')
    opt_parser.add_argument('--optimizer', default='LBFGS', choices=['LBFGS', 'BFGS', 'FIRE'],
                           help='Optimization algorithm')
    opt_parser.add_argument('--cell', action='store_true',
                           help='Optimize cell parameters')
    opt_parser.add_argument('--output', type=str, help='Output structure file')
    opt_parser.add_argument('--trajectory', type=str, help='Trajectory file path')
    
    # ==================== Molecular Dynamics ====================
    md_parser = subparsers.add_parser(
        'md',
        help='Run molecular dynamics simulation'
    )
    md_parser.add_argument('structure', type=str, help='Structure file path')
    md_parser.add_argument('--model', default='grace-2l', help='GRACE model name')
    md_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                          help='Computing device')
    md_parser.add_argument('--ensemble', default='nvt', choices=['nve', 'nvt', 'npt'],
                          help='MD ensemble')
    md_parser.add_argument('--temp', default=300, type=float,
                          help='Temperature (K)')
    md_parser.add_argument('--pressure', type=float,
                          help='Pressure for NPT (GPa)')
    md_parser.add_argument('--steps', default=1000, type=int,
                          help='Number of MD steps')
    md_parser.add_argument('--timestep', default=1.0, type=float,
                          help='Time step (fs)')
    md_parser.add_argument('--trajectory', type=str, help='Trajectory file path')
    md_parser.add_argument('--logfile', type=str, help='Log file path')
    
    # ==================== Phonon Calculation ====================
    phonon_parser = subparsers.add_parser(
        'phonon',
        help='Calculate phonon properties'
    )
    phonon_parser.add_argument('structure', type=str, help='Structure file path')
    phonon_parser.add_argument('--model', default='grace-2l', help='GRACE model name')
    phonon_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                              help='Computing device')
    phonon_parser.add_argument('--supercell', nargs=3, type=int, default=[2, 2, 2],
                              help='Supercell size (nx ny nz)')
    phonon_parser.add_argument('--mesh', nargs=3, type=int, default=[20, 20, 20],
                              help='k-point mesh (nx ny nz)')
    phonon_parser.add_argument('--temp-range', nargs=3, type=float,
                              help='Temperature range (min max step) in K')
    phonon_parser.add_argument('--output-dir', type=str, help='Output directory')
    
    # ==================== Bulk Modulus ====================
    bulk_parser = subparsers.add_parser(
        'bulk-modulus',
        help='Calculate bulk modulus'
    )
    bulk_parser.add_argument('structure', type=str, help='Structure file path')
    bulk_parser.add_argument('--model', default='grace-2l', help='GRACE model name')
    bulk_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                            help='Computing device')
    bulk_parser.add_argument('--points', default=11, type=int,
                            help='Number of volume points')
    bulk_parser.add_argument('--strain-range', default=0.05, type=float,
                            help='Strain range (±)')
    bulk_parser.add_argument('--eos', default='birchmurnaghan',
                            choices=['birchmurnaghan', 'vinet', 'murnaghan', 'sjeos', 'taylor'],
                            help='Equation of state type')
    bulk_parser.add_argument('--plot', action='store_true',
                            help='Generate EOS plot')
    bulk_parser.add_argument('--output-dir', type=str, help='Output directory')
    
    # ==================== Adsorption Energy ====================
    ads_parser = subparsers.add_parser(
        'adsorption',
        help='Calculate adsorption energy'
    )
    ads_parser.add_argument('host', type=str, help='Host structure file (e.g., MOF)')
    ads_parser.add_argument('adsorbate', type=str, help='Adsorbate structure file (e.g., CO2)')
    ads_parser.add_argument('combined', type=str, help='Combined structure file')
    ads_parser.add_argument('--model', default='grace-2l', help='GRACE model name')
    ads_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                           help='Computing device')
    ads_parser.add_argument('--relax-all', action='store_true',
                           help='Relax all structures (host, adsorbate, combined)')
    ads_parser.add_argument('--no-relax-host', action='store_true',
                           help='Do not relax host structure')
    ads_parser.add_argument('--no-relax-adsorbate', action='store_true',
                           help='Do not relax adsorbate structure')
    ads_parser.add_argument('--no-relax-combined', action='store_true',
                           help='Do not relax combined structure')
    ads_parser.add_argument('--fmax', default=0.05, type=float,
                           help='Force convergence for relaxation (eV/Å)')
    ads_parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    
    # ==================== Device Info ====================
    info_parser = subparsers.add_parser(
        'device-info',
        help='Show device information'
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Import here to avoid slow startup
    from grace_inference import GRACEInference
    from grace_inference.utils import print_device_info, read_structure, write_structure
    
    # Handle device-info command
    if args.command == 'device-info':
        print_device_info()
        return
    
    # Initialize GRACE calculator
    print(f"Loading GRACE model: {args.model}")
    print(f"Device: {args.device}")
    
    try:
        calc = GRACEInference(
            model_name=args.model,
            device=args.device
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # ==================== Execute Command ====================
    
    if args.command == 'single-point':
        # Single-point calculation
        print(f"\nReading structure: {args.structure}")
        result = calc.single_point(args.structure)
        
        print("\n" + "=" * 60)
        print("Single-Point Calculation Results")
        print("=" * 60)
        print(f"Energy: {result['energy']:.6f} eV")
        print(f"Energy per atom: {result['energy_per_atom']:.6f} eV/atom")
        print(f"Max force: {result['max_force']:.6f} eV/Å")
        print(f"RMS force: {result['rms_force']:.6f} eV/Å")
        
        if result.get('pressure_GPa') is not None:
            print(f"Pressure: {result['pressure_GPa']:.4f} GPa")
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                output_dict = {k: v.tolist() if hasattr(v, 'tolist') else v 
                              for k, v in result.items()}
                json.dump(output_dict, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    elif args.command == 'optimize':
        # Structure optimization
        print(f"\nOptimizing structure: {args.structure}")
        
        optimized = calc.optimize(
            args.structure,
            fmax=args.fmax,
            steps=args.steps,
            optimizer=args.optimizer,
            optimize_cell=args.cell,
            trajectory=args.trajectory,
            output=args.output
        )
        
        print("\n" + "=" * 60)
        print("Optimization Complete")
        print("=" * 60)
        
        # Calculate final energy
        final_result = calc.single_point(optimized)
        print(f"Final energy: {final_result['energy']:.6f} eV")
        print(f"Final max force: {final_result['max_force']:.6f} eV/Å")
        
        if args.output:
            print(f"Optimized structure saved to {args.output}")
    
    elif args.command == 'md':
        # Molecular dynamics
        print(f"\nRunning MD simulation: {args.structure}")
        print(f"Ensemble: {args.ensemble}")
        print(f"Temperature: {args.temp} K")
        if args.pressure is not None:
            print(f"Pressure: {args.pressure} GPa")
        print(f"Steps: {args.steps}")
        print(f"Timestep: {args.timestep} fs")
        
        final_atoms = calc.molecular_dynamics(
            args.structure,
            ensemble=args.ensemble,
            temperature_K=args.temp,
            pressure_GPa=args.pressure,
            timestep=args.timestep,
            steps=args.steps,
            trajectory=args.trajectory,
            logfile=args.logfile
        )
        
        print("\n" + "=" * 60)
        print("MD Simulation Complete")
        print("=" * 60)
        
        if args.trajectory:
            print(f"Trajectory saved to {args.trajectory}")
    
    elif args.command == 'phonon':
        # Phonon calculation
        print(f"\nCalculating phonon properties: {args.structure}")
        print(f"Supercell: {args.supercell}")
        print(f"k-point mesh: {args.mesh}")
        
        temp_range = tuple(args.temp_range) if args.temp_range else None
        
        result = calc.phonon(
            args.structure,
            supercell=tuple(args.supercell),
            mesh=tuple(args.mesh),
            temperature_range=temp_range,
            output_dir=args.output_dir
        )
        
        print("\n" + "=" * 60)
        print("Phonon Calculation Complete")
        print("=" * 60)
        
        if args.output_dir:
            print(f"Results saved to {args.output_dir}")
    
    elif args.command == 'bulk-modulus':
        # Bulk modulus calculation
        print(f"\nCalculating bulk modulus: {args.structure}")
        print(f"Strain range: ±{args.strain_range}")
        print(f"Number of points: {args.points}")
        print(f"EOS type: {args.eos}")
        
        result = calc.bulk_modulus(
            args.structure,
            num_points=args.points,
            strain_range=args.strain_range,
            eos=args.eos,
            plot=args.plot,
            output_dir=args.output_dir
        )
        
        if 'error' not in result:
            print("\n" + "=" * 60)
            print("Bulk Modulus Calculation Complete")
            print("=" * 60)
            print(f"Bulk modulus: {result['bulk_modulus_GPa']:.2f} GPa")
            print(f"Equilibrium volume: {result['v0']:.3f} Å³")
    
    elif args.command == 'adsorption':
        # Adsorption energy calculation
        print(f"\nCalculating adsorption energy")
        print(f"Host: {args.host}")
        print(f"Adsorbate: {args.adsorbate}")
        print(f"Combined: {args.combined}")
        
        # Determine relaxation flags
        relax_host = not args.no_relax_host if not args.relax_all else True
        relax_adsorbate = not args.no_relax_adsorbate if not args.relax_all else True
        relax_combined = not args.no_relax_combined if not args.relax_all else True
        
        if args.relax_all:
            relax_host = relax_adsorbate = relax_combined = True
        
        result = calc.adsorption_energy(
            args.host,
            args.adsorbate,
            args.combined,
            relax_host=relax_host,
            relax_adsorbate=relax_adsorbate,
            relax_combined=relax_combined,
            fmax=args.fmax
        )
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                output_dict = {
                    'adsorption_energy_eV': result['adsorption_energy_eV'],
                    'adsorption_energy_kJ_mol': result['adsorption_energy_kJ_mol'],
                    'host_energy': result['host_energy'],
                    'adsorbate_energy': result['adsorbate_energy'],
                    'combined_energy': result['combined_energy'],
                }
                json.dump(output_dict, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
