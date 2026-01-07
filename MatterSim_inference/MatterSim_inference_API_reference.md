# MatterSim Inference - API Reference

> **MatterSim**: MOFSimBench 排名 **#3** 的通用机器学习力场  
> **特色**: 吸附能 #1、MD 稳定性 #1、不确定性估计

---

## 目录

1. [核心类: MatterSimInference](#1-核心类-mattersim_inference)
2. [Utils 模块](#2-utils-模块)
3. [Tasks 模块](#3-tasks-模块)
4. [CLI 命令行工具](#4-cli-命令行工具)

---

## 1. 核心类: MatterSimInference

### 1.1 类定义

```python
class MatterSimInference:
    """
    MatterSim 推理主类。
    
    封装 MatterSim 模型，提供材料性质计算的统一接口。
    
    Attributes:
        model_name: 模型名称
        device: 计算设备 ('cuda' / 'cpu')
        calculator: ASE Calculator 实例
    """
```

### 1.2 初始化

```python
def __init__(
    self,
    model_name: str = "MatterSim-v1-5M",
    device: str = "auto",
    **kwargs
) -> None:
    """
    初始化 MatterSimInference。
    
    Args:
        model_name: 模型名称
            - "MatterSim-v1-1M": 1M 参数轻量版
            - "MatterSim-v1-5M": 5M 参数标准版 (默认)
        device: 计算设备
            - "auto": 自动检测 (优先 GPU)
            - "cuda": 强制使用 GPU
            - "cpu": 强制使用 CPU
        **kwargs: 传递给 MatterSimCalculator 的额外参数
    
    Example:
        >>> calc = MatterSimInference(model_name="MatterSim-v1-5M", device="cuda")
        >>> print(calc)
        MatterSimInference(model=MatterSim-v1-5M, device=cuda)
    """
```

### 1.3 方法

#### 1.3.1 single_point

```python
def single_point(
    self,
    atoms: Union[Atoms, str, Path]
) -> Dict[str, Any]:
    """
    单点能量计算。
    
    Args:
        atoms: ASE Atoms 对象或结构文件路径
    
    Returns:
        dict: 包含以下键:
            - energy: 总能量 (eV)
            - energy_per_atom: 每原子能量 (eV/atom)
            - forces: 力数组 (N, 3) (eV/Å)
            - stress: 应力张量 (6,) (eV/Å³)
            - max_force: 最大力分量 (eV/Å)
            - rms_force: RMS 力 (eV/Å)
            - pressure: 压强 (GPa)
    
    Example:
        >>> result = calc.single_point("MOF-5.cif")
        >>> print(f"Energy: {result['energy']:.4f} eV")
    """
```

#### 1.3.2 optimize

```python
def optimize(
    self,
    atoms: Union[Atoms, str, Path],
    fmax: float = 0.05,
    optimizer: str = "LBFGS",
    optimize_cell: bool = False,
    max_steps: int = 500,
    output: Optional[str] = None
) -> Dict[str, Any]:
    """
    结构优化。
    
    Args:
        atoms: ASE Atoms 对象或结构文件路径
        fmax: 力收敛阈值 (eV/Å)
        optimizer: 优化器类型 ("LBFGS", "BFGS", "FIRE")
        optimize_cell: 是否同时优化晶胞
        max_steps: 最大优化步数
        output: 输出文件路径 (可选)
    
    Returns:
        dict: 包含以下键:
            - converged: 是否收敛
            - steps: 优化步数
            - initial_energy: 初始能量 (eV)
            - final_energy: 最终能量 (eV)
            - energy_change: 能量变化 (eV)
            - final_fmax: 最终最大力 (eV/Å)
            - atoms: 优化后的 Atoms 对象
    
    Example:
        >>> result = calc.optimize("MOF-5.cif", fmax=0.01, optimize_cell=True)
        >>> print(f"Converged: {result['converged']}")
    """
```

#### 1.3.3 run_md

```python
def run_md(
    self,
    atoms: Union[Atoms, str, Path],
    ensemble: str = "nvt",
    temperature: float = 300.0,
    pressure: Optional[float] = None,
    steps: int = 10000,
    timestep: float = 1.0,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100
) -> Atoms:
    """
    分子动力学模拟。
    
    Args:
        atoms: ASE Atoms 对象或结构文件路径
        ensemble: 系综类型 ("nve", "nvt", "npt")
        temperature: 温度 (K)
        pressure: 压强 (GPa), 仅 NPT 需要
        steps: 模拟步数
        timestep: 时间步长 (fs)
        trajectory: 轨迹文件路径
        logfile: 日志文件路径
        log_interval: 日志记录间隔
    
    Returns:
        Atoms: 最终结构
    
    Example:
        >>> final = calc.run_md(atoms, ensemble="nvt", temperature=300, steps=50000)
    """
```

#### 1.3.4 phonon

```python
def phonon(
    self,
    atoms: Union[Atoms, str, Path],
    supercell_matrix: List[int] = [2, 2, 2],
    mesh: List[int] = [20, 20, 20],
    displacement: float = 0.01,
    t_min: float = 0,
    t_max: float = 1000,
    t_step: float = 10
) -> Dict[str, Any]:
    """
    声子计算。
    
    Args:
        atoms: 原胞结构
        supercell_matrix: 超胞大小 [a, b, c]
        mesh: k 点网格 [kx, ky, kz]
        displacement: 原子位移 (Å)
        t_min: 最低温度 (K)
        t_max: 最高温度 (K)
        t_step: 温度步长 (K)
    
    Returns:
        dict: 包含以下键:
            - frequency_points: 频率点 (THz)
            - total_dos: 态密度
            - has_imaginary: 是否有虚频
            - imaginary_modes: 虚频数量
            - thermal: 热力学性质字典
                - temperatures: 温度数组 (K)
                - heat_capacity: 热容 (J/(mol·K))
                - entropy: 熵 (J/(mol·K))
                - free_energy: 自由能 (kJ/mol)
    
    Example:
        >>> result = calc.phonon(primitive, supercell_matrix=[3, 3, 3])
        >>> print(f"Imaginary modes: {result['imaginary_modes']}")
    """
```

#### 1.3.5 bulk_modulus

```python
def bulk_modulus(
    self,
    atoms: Union[Atoms, str, Path],
    strain_range: float = 0.05,
    npoints: int = 11,
    eos: str = "birchmurnaghan"
) -> Dict[str, Any]:
    """
    体模量计算。
    
    Args:
        atoms: ASE Atoms 对象或结构文件路径
        strain_range: 应变范围 (±)
        npoints: 采样点数
        eos: 状态方程类型
            - "birchmurnaghan": Birch-Murnaghan (默认)
            - "vinet": Vinet EOS
            - "murnaghan": Murnaghan EOS
    
    Returns:
        dict: 包含以下键:
            - bulk_modulus: 体模量 (GPa)
            - v0: 平衡体积 (Å³)
            - e0: 平衡能量 (eV)
            - eos: 使用的 EOS 类型
            - volumes: 体积数组
            - energies: 能量数组
    
    Example:
        >>> result = calc.bulk_modulus(atoms, strain_range=0.05)
        >>> print(f"Bulk modulus: {result['bulk_modulus']:.2f} GPa")
    """
```

#### 1.3.6 adsorption_energy

```python
def adsorption_energy(
    self,
    mof_structure: Union[Atoms, str, Path],
    gas_molecule: str,
    site_position: List[float],
    optimize_complex: bool = True,
    fmax: float = 0.05
) -> Dict[str, Any]:
    """
    吸附能计算。
    
    MatterSim 在此任务上表现 **最佳** (#1 in MOFSimBench)。
    
    Args:
        mof_structure: MOF 结构
        gas_molecule: 气体分子名称 ("CO2", "H2O", "CH4", ...)
        site_position: 吸附位点坐标 [x, y, z]
        optimize_complex: 是否优化复合体
        fmax: 优化收敛阈值 (eV/Å)
    
    Returns:
        dict: 包含以下键:
            - E_ads: 吸附能 (eV)
            - E_mof: MOF 能量 (eV)
            - E_gas: 气体分子能量 (eV)
            - E_complex: 复合体能量 (eV)
            - complex_atoms: 复合体结构
    
    Example:
        >>> result = calc.adsorption_energy(mof, "CO2", [10, 10, 10])
        >>> print(f"E_ads: {result['E_ads']:.4f} eV")
    """
```

#### 1.3.7 coordination

```python
def coordination(
    self,
    atoms: Union[Atoms, str, Path],
    cutoff: float = 3.0
) -> Dict[str, Any]:
    """
    配位环境分析。
    
    Args:
        atoms: ASE Atoms 对象或结构文件路径
        cutoff: 配位判断截断距离 (Å)
    
    Returns:
        dict: 包含以下键:
            - coordination: 各金属原子配位信息
                - coordination_number: 配位数
                - neighbors: 配位原子列表
                - average_distance: 平均键长
            - metal_indices: 金属原子索引
    
    Example:
        >>> result = calc.coordination(mof, cutoff=3.0)
        >>> for idx, info in result['coordination'].items():
        ...     print(f"Metal {idx}: CN = {info['coordination_number']}")
    """
```

---

## 2. Utils 模块

### 2.1 device.py

```python
def get_device(device: str = "auto") -> str:
    """
    获取计算设备。
    
    Args:
        device: 设备选择
            - "auto": 自动检测
            - "cuda": 强制 GPU
            - "cpu": 强制 CPU
    
    Returns:
        str: 实际使用的设备
    """

def get_available_devices() -> List[str]:
    """返回可用设备列表。"""

def print_device_info() -> None:
    """打印设备信息。"""
```

### 2.2 io.py

```python
def read_structure(filepath: Union[str, Path]) -> Atoms:
    """
    读取结构文件。
    
    支持格式: CIF, POSCAR, XYZ, etc.
    """

def write_structure(atoms: Atoms, filepath: Union[str, Path], **kwargs) -> None:
    """写入结构文件。"""

def validate_structure(atoms: Atoms) -> Tuple[bool, str]:
    """验证结构有效性。"""
```

---

## 3. Tasks 模块

### 3.1 static.py

```python
def calculate_single_point(
    atoms: Atoms,
    calculator: Any
) -> Dict[str, Any]:
    """执行单点计算。"""
```

### 3.2 dynamics.py

```python
def run_md(
    atoms: Atoms,
    calculator: Any,
    ensemble: str,
    temperature: float,
    **kwargs
) -> Atoms:
    """运行分子动力学。"""

def analyze_md_trajectory(trajectory: List[Atoms]) -> Dict[str, Any]:
    """分析 MD 轨迹。"""
```

### 3.3 phonon.py

```python
def calculate_phonon(
    atoms: Atoms,
    calculator: Any,
    supercell_matrix: List[int],
    mesh: List[int],
    **kwargs
) -> Dict[str, Any]:
    """计算声子性质。"""
```

### 3.4 mechanics.py

```python
def calculate_bulk_modulus(
    atoms: Atoms,
    calculator: Any,
    strain_range: float,
    npoints: int,
    eos: str
) -> Dict[str, Any]:
    """计算体模量。"""
```

### 3.5 adsorption.py

```python
def calculate_adsorption_energy(
    mof: Atoms,
    gas: str,
    site: List[float],
    calculator: Any,
    **kwargs
) -> Dict[str, Any]:
    """计算吸附能。"""
```

---

## 4. CLI 命令行工具

### 4.1 命令概览

```bash
mattersim-infer <command> [options]
```

### 4.2 可用命令

| 命令 | 说明 |
|------|------|
| `single-point` | 单点能量计算 |
| `optimize` | 结构优化 |
| `md` | 分子动力学 |
| `phonon` | 声子计算 |
| `bulk-modulus` | 体模量计算 |
| `adsorption` | 吸附能计算 |
| `batch-optimize` | 批量优化 |

### 4.3 命令示例

```bash
# 单点计算
mattersim-infer single-point MOF-5.cif --output result.json

# 结构优化
mattersim-infer optimize MOF-5.cif --fmax 0.01 --cell --output optimized.cif

# 分子动力学
mattersim-infer md MOF-5.cif --ensemble nvt --temp 300 --steps 50000

# 声子计算
mattersim-infer phonon primitive.cif --supercell 2 2 2 --mesh 20 20 20

# 体模量
mattersim-infer bulk-modulus MOF-5.cif --strain-range 0.05

# 吸附能
mattersim-infer adsorption MOF.cif --gas CO2 --site 10 10 10

# 批量优化
mattersim-infer batch-optimize mof_database/*.cif --output-dir optimized/
```

### 4.4 全局选项

| 选项 | 说明 |
|------|------|
| `--model` | 模型选择 (MatterSim-v1-1M, MatterSim-v1-5M) |
| `--device` | 设备选择 (auto, cuda, cpu) |
| `--output` | 输出文件 |
| `--verbose` | 详细输出 |

---

## 附录

### A. 返回值类型汇总

| 方法 | 返回类型 | 主要键 |
|------|----------|--------|
| `single_point` | Dict | energy, forces, stress |
| `optimize` | Dict | converged, atoms, final_energy |
| `run_md` | Atoms | 最终结构 |
| `phonon` | Dict | frequency_points, thermal |
| `bulk_modulus` | Dict | bulk_modulus, v0, e0 |
| `adsorption_energy` | Dict | E_ads, E_mof, E_gas |
| `coordination` | Dict | coordination |

### B. 错误处理

```python
from mattersim_inference import MatterSimInference
from mattersim_inference.exceptions import (
    ModelNotFoundError,
    StructureError,
    ConvergenceError
)

try:
    calc = MatterSimInference()
    result = calc.optimize(atoms, fmax=0.01)
except ModelNotFoundError:
    print("模型未找到")
except ConvergenceError:
    print("优化未收敛")
```

---

**Last Updated**: 2026-01-07  
**Version**: 0.1.0
