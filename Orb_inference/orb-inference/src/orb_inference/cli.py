"""Command-line interface for orb-inference."""

import click
from pathlib import Path
import sys

from orb_inference import OrbInference
from orb_inference.utils.io import load_structure, save_structure


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    Orb Inference CLI - Materials science calculations with Orb models.
    
    Examples:
    
        # Single-point energy
        orb-infer energy structure.cif --model orb-v3-omat
        
        # Optimize structure  
        orb-infer optimize structure.cif -o optimized.cif --fmax 0.01
        
        # Molecular dynamics
        orb-infer md structure.cif -T 300 -n 5000 --ensemble nvt
        
        # Phonon calculation
        orb-infer phonon structure.cif --supercell 2,2,2
    """
    pass


@cli.command()
@click.argument('structure', type=click.Path(exists=True))
@click.option('--model', '-m', default='orb-v3-omat', 
              help='Orb model name')
@click.option('--device', '-d', default=None,
              help='Device (cuda/cpu/mps)')
@click.option('--precision', '-p', default='float32-high',
              help='Model precision')
def energy(structure, model, device, precision):
    """Calculate single-point energy."""
    click.echo(f"Loading model: {model}")
    orb = OrbInference(model_name=model, device=device, precision=precision)
    
    click.echo(f"Calculating energy for: {structure}")
    result = orb.single_point(structure)
    
    click.echo("\n=== Results ===")
    click.echo(f"Energy: {result['energy']:.6f} eV")
    click.echo(f"Max force: {result['max_force']:.4f} eV/Å")
    click.echo(f"RMS force: {result['rms_force']:.4f} eV/Å")
    
    if 'stress' in result:
        click.echo(f"Pressure: {result['pressure']:.4f} GPa")


@cli.command()
@click.argument('structure', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output structure path')
@click.option('--fmax', '-f', default=0.05, type=float,
              help='Force convergence (eV/Å)')
@click.option('--optimizer', default='LBFGS',
              type=click.Choice(['LBFGS', 'BFGS', 'FIRE']),
              help='Optimizer type')
@click.option('--relax-cell', is_flag=True,
              help='Relax cell vectors')
@click.option('--model', '-m', default='orb-v3-omat',
              help='Orb model name')
@click.option('--device', '-d', default=None,
              help='Device (cuda/cpu/mps)')
def optimize(structure, output, fmax, optimizer, relax_cell, model, device):
    """Optimize structure."""
    click.echo(f"Loading model: {model}")
    orb = OrbInference(model_name=model, device=device)
    
    click.echo(f"Optimizing: {structure}")
    click.echo(f"  fmax: {fmax} eV/Å")
    click.echo(f"  optimizer: {optimizer}")
    click.echo(f"  relax_cell: {relax_cell}")
    
    result = orb.optimize(
        structure, 
        fmax=fmax,
        optimizer=optimizer,
        relax_cell=relax_cell
    )
    
    click.echo("\n=== Results ===")
    click.echo(f"Converged: {result['converged']}")
    click.echo(f"Steps: {result['steps']}")
    click.echo(f"Final energy: {result['final_energy']:.6f} eV")
    click.echo(f"Final fmax: {result['final_fmax']:.6f} eV/Å")
    
    if output:
        save_structure(result['atoms'], output)
        click.echo(f"\nOptimized structure saved to: {output}")
    else:
        click.echo("\nUse --output to save optimized structure")


@cli.command()
@click.argument('structure', type=click.Path(exists=True))
@click.option('--temperature', '-T', default=300.0, type=float,
              help='Temperature (K)')
@click.option('--steps', '-n', default=1000, type=int,
              help='Number of MD steps')
@click.option('--timestep', '-dt', default=1.0, type=float,
              help='Timestep (fs)')
@click.option('--ensemble', default='nvt',
              type=click.Choice(['nvt', 'npt']),
              help='MD ensemble')
@click.option('--pressure', '-P', type=float,
              help='Pressure for NPT (GPa)')
@click.option('--trajectory', '-t', type=click.Path(),
              help='Trajectory output file')
@click.option('--logfile', '-l', type=click.Path(),
              help='Log file path')
@click.option('--model', '-m', default='orb-v3-omat',
              help='Orb model name')
@click.option('--device', '-d', default=None,
              help='Device (cuda/cpu/mps)')
def md(structure, temperature, steps, timestep, ensemble, pressure, 
       trajectory, logfile, model, device):
    """Run molecular dynamics."""
    click.echo(f"Loading model: {model}")
    orb = OrbInference(model_name=model, device=device)
    
    click.echo(f"Running {ensemble.upper()} MD on: {structure}")
    click.echo(f"  Temperature: {temperature} K")
    click.echo(f"  Steps: {steps}")
    click.echo(f"  Timestep: {timestep} fs")
    if ensemble == 'npt':
        if pressure is None:
            click.echo("Error: Must specify --pressure for NPT", err=True)
            sys.exit(1)
        click.echo(f"  Pressure: {pressure} GPa")
    
    final_atoms = orb.run_md(
        structure,
        temperature=temperature,
        steps=steps,
        timestep=timestep,
        ensemble=ensemble,
        pressure=pressure,
        trajectory=trajectory,
        logfile=logfile
    )
    
    click.echo(f"\n=== MD Completed ===")
    click.echo(f"Final temperature: {final_atoms.get_temperature():.2f} K")
    
    if trajectory:
        click.echo(f"Trajectory saved to: {trajectory}")
    if logfile:
        click.echo(f"Log saved to: {logfile}")


@cli.command()
@click.argument('structure', type=click.Path(exists=True))
@click.option('--supercell', '-s', default='2,2,2',
              help='Supercell matrix (e.g., 2,2,2)')
@click.option('--mesh', default='20,20,20',
              help='k-point mesh (e.g., 20,20,20)')
@click.option('--model', '-m', default='orb-v3-omat',
              help='Orb model name')
@click.option('--device', '-d', default=None,
              help='Device (cuda/cpu/mps)')
def phonon(structure, supercell, mesh, model, device):
    """Calculate phonon properties."""
    click.echo(f"Loading model: {model}")
    orb = OrbInference(model_name=model, device=device)
    
    # Parse supercell and mesh
    supercell_matrix = [int(x) for x in supercell.split(',')]
    mesh_points = [int(x) for x in mesh.split(',')]
    
    click.echo(f"Calculating phonon for: {structure}")
    click.echo(f"  Supercell: {supercell_matrix}")
    click.echo(f"  Mesh: {mesh_points}")
    
    result = orb.phonon(
        structure,
        supercell_matrix=supercell_matrix,
        mesh=mesh_points
    )
    
    # Find zero-point energy contribution (acoustic modes at Γ)
    freq = result['frequency_points']
    idx_positive = freq > 0.1  # Skip near-zero modes
    
    click.echo("\n=== Phonon Results ===")
    click.echo(f"Frequency range: {freq[idx_positive].min():.2f} - {freq.max():.2f} THz")
    
    # Print heat capacity at selected temperatures
    temps = result['thermal']['temperatures']
    Cv = result['thermal']['heat_capacity']
    
    click.echo("\nHeat Capacity [J/(K·mol)]:")
    for T_target in [100, 200, 300, 400, 500]:
        idx = (temps >= T_target).argmax()
        if temps[idx] <= T_target + 5:
            click.echo(f"  {temps[idx]:.0f} K: {Cv[idx]:.2f}")


@cli.command(name='bulk-modulus')
@click.argument('structure', type=click.Path(exists=True))
@click.option('--strain', default=0.05, type=float,
              help='Volume strain range')
@click.option('--points', '-n', default=7, type=int,
              help='Number of volume points')
@click.option('--optimize/--no-optimize', default=True,
              help='Optimize before EOS calculation')
@click.option('--model', '-m', default='orb-v3-omat',
              help='Orb model name')
@click.option('--device', '-d', default=None,
              help='Device (cuda/cpu/mps)')
def bulk_modulus_cmd(structure, strain, points, optimize, model, device):
    """Calculate bulk modulus."""
    click.echo(f"Loading model: {model}")
    orb = OrbInference(model_name=model, device=device)
    
    click.echo(f"Calculating bulk modulus for: {structure}")
    click.echo(f"  Strain range: ±{strain*100:.1f}%")
    click.echo(f"  Volume points: {points}")
    
    result = orb.bulk_modulus(
        structure,
        strain_range=strain,
        n_points=points,
        optimize_first=optimize
    )
    
    click.echo("\n=== Results ===")
    click.echo(f"Bulk modulus: {result['bulk_modulus']:.2f} GPa")
    click.echo(f"Equilibrium volume: {result['equilibrium_volume']:.3f} Å³")
    click.echo(f"Equilibrium energy: {result['equilibrium_energy']:.6f} eV")


@cli.command()
@click.option('--host', type=click.Path(exists=True), required=True,
              help='Host structure (MOF)')
@click.option('--guest', type=click.Path(exists=True), required=True,
              help='Guest molecule')
@click.option('--complex', 'complex_path', type=click.Path(exists=True), required=True,
              help='Host+guest complex')
@click.option('--optimize/--no-optimize', default=True,
              help='Optimize complex before calculation')
@click.option('--model', '-m', default='orb-v3-omat',
              help='Orb model name')
@click.option('--device', '-d', default=None,
              help='Device (cuda/cpu/mps)')
def adsorption(host, guest, complex_path, optimize, model, device):
    """Calculate adsorption energy."""
    click.echo(f"Loading model: {model}")
    orb = OrbInference(model_name=model, device=device)
    
    click.echo(f"Calculating adsorption energy:")
    click.echo(f"  Host: {host}")
    click.echo(f"  Guest: {guest}")
    click.echo(f"  Complex: {complex_path}")
    
    result = orb.adsorption_energy(
        host=host,
        guest=guest,
        complex_atoms=complex_path,
        optimize_complex=optimize
    )
    
    click.echo("\n=== Results ===")
    click.echo(f"E_ads: {result['E_ads']:.6f} eV")
    click.echo(f"E_ads per atom: {result['E_ads_per_atom']:.6f} eV/atom")
    
    if result['E_ads'] < 0:
        click.echo("→ Stable adsorption (E_ads < 0)")
    else:
        click.echo("→ Unstable adsorption (E_ads > 0)")


@cli.command()
@click.option('--model', '-m', default='orb-v3-omat',
              help='Orb model name')
@click.option('--device', '-d', default=None,
              help='Device (cuda/cpu/mps)')
def info(model, device):
    """Show model and device information."""
    orb = OrbInference(model_name=model, device=device)
    info_dict = orb.info()
    
    click.echo("\n=== Orb Inference Info ===")
    click.echo(f"Model: {info_dict['model_name']}")
    click.echo(f"Device: {info_dict['device']}")
    click.echo(f"Precision: {info_dict['precision']}")
    
    if 'device_info' in info_dict:
        dev_info = info_dict['device_info']
        click.echo(f"\nDevice Details:")
        click.echo(f"  Name: {dev_info.get('name', 'N/A')}")
        if 'memory_total' in dev_info:
            click.echo(f"  Memory: {dev_info['memory_allocated']:.2f} / "
                      f"{dev_info['memory_total']:.2f} GB")


if __name__ == '__main__':
    cli()
