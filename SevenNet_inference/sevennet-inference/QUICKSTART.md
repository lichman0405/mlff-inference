# SevenNet Inference Quick Start

> Get started with SevenNet inference in 5 minutes

## 1. Installation

```bash
pip install sevennet-inference
```

## 2. Verify Installation

```bash
sevennet-infer --help
```

## 3. First Calculation

### Python

```python
from sevennet_inference import SevenNetInference

# Initialize
calc = SevenNetInference(model_name="SevenNet-0", device="auto")

# Single-point calculation
result = calc.single_point("your_structure.cif")
print(f"Energy: {result['energy']:.4f} eV")
print(f"Max Force: {result['max_force']:.4f} eV/Å")
```

### Command Line

```bash
sevennet-infer single-point your_structure.cif
```

## 4. Core Features

### 4.1 Single-Point Calculation

```python
result = calc.single_point(atoms)
# Returns: energy, forces, stress, pressure
```

### 4.2 Structure Optimization

```python
result = calc.optimize(atoms, fmax=0.05, optimize_cell=True)
# Returns: converged, atoms, final_energy
```

### 4.3 Molecular Dynamics

```python
final = calc.run_md(atoms, ensemble="nvt", temperature=300, steps=10000)
```

### 4.4 Phonon Calculation

```python
result = calc.calculate_phonon(atoms, supercell=[2, 2, 2])
print(f"Zero-point Energy: {result['ZPE']:.4f} eV")
```

## 5. Common Options

| Option | Description |
|------|------|
| `model_name` | SevenNet-0, SevenNet-0-22May2024 |
| `device` | auto, cuda, cpu |
| `fmax` | Force convergence threshold (eV/Å) |

## 6. Next Steps

- Check [examples/](examples/) for more examples
- Read [Full Documentation](../SevenNet_inference_tasks.md)
