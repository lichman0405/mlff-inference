# EquiformerV2 Inference API 参考文档

> **版本**: 0.1.0  
> **更新日期**: 2026年1月7日

本文档提供了 EquiformerV2 Inference 包的完整 API 参考，包括核心类、工具函数、任务模块和命令行接口。

---

## 目录

1. [核心模块](#核心模块)
   - [EquiformerV2Inference 类](#equiformerv2inference-类)
2. [工具模块 (Utils)](#工具模块-utils)
   - [设备管理](#设备管理)
   - [输入输出](#输入输出)
3. [任务模块 (Tasks)](#任务模块-tasks)
   - [静态计算](#静态计算)
   - [分子动力学](#分子动力学)
   - [声子计算](#声子计算)
   - [力学性质](#力学性质)
4. [命令行接口 (CLI)](#命令行接口-cli)

---

## 核心模块

### EquiformerV2Inference 类

**完整路径**: `equiformerv2_inference.EquiformerV2Inference`

EquiformerV2 推理计算器的主类，提供了统一的接口进行能量、力和应力计算，以及结构优化、分子动力学等任务。

#### 类初始化

```python
EquiformerV2Inference(
    model_name: str = "EquiformerV2-31M-S2EF",
    model_path: Optional[str] = None,
    device: str = "auto",
    dtype: str = "float32"
)
```

**参数**:
- `model_name` (str): 模型名称
  - 可选值:
    - `"EquiformerV2-31M-S2EF"` - 3100万参数的 S2EF 模型（默认）
    - `"EquiformerV2-153M-S2EF"` - 1.53亿参数的大模型，精度更高
    - `"equiformer_v2_31M"` - 31M 模型的别名
    - `"equiformer_v2_153M"` - 153M 模型的别名
- `model_path` (Optional[str]): 自定义模型检查点路径（可选）
- `device` (str): 计算设备
  - `"auto"` - 自动选择（优先使用 GPU）
  - `"cuda"` - 强制使用 GPU
  - `"cpu"` - 强制使用 CPU
  - `"cuda:0"`, `"cuda:1"` 等 - 指定 GPU 设备
- `dtype` (str): 数据类型，可选 `"float32"` 或 `"float64"`

**返回值**: EquiformerV2Inference 实例

**异常**:
- `ImportError`: 当 Open Catalyst Project (ocp) 未安装时
- `ValueError`: 当请求 CUDA 但 CUDA 不可用时

**示例**:

```python
from equiformerv2_inference import EquiformerV2Inference

# 自动选择设备，使用默认模型
calc = EquiformerV2Inference()

# 使用大模型和指定设备
calc = EquiformerV2Inference(
    model_name="EquiformerV2-153M-S2EF",
    device="cuda:0"
)

# 从自定义检查点加载
calc = EquiformerV2Inference(
    model_path="/path/to/checkpoint.pt",
    device="cuda"
)
```

---

#### 方法 1: single_point()

单点能量和力计算。

```python
calc.single_point(
    structure: Union[str, Path, Atoms],
    properties: List[str] = ["energy", "forces", "stress"]
) -> Dict[str, Any]
```

**参数**:
- `structure` (Union[str, Path, Atoms]): 结构文件路径或 ASE Atoms 对象
  - 支持格式: CIF, POSCAR, XYZ, VASP, Quantum Espresso 等
- `properties` (List[str]): 要计算的性质列表
  - `"energy"` - 总能量
  - `"forces"` - 原子受力
  - `"stress"` - 应力张量

**返回值**: Dict[str, Any]
- `energy` (float): 总能量 (eV)
- `energy_per_atom` (float): 每原子能量 (eV/atom)
- `forces` (ndarray): 力数组，形状为 (n_atoms, 3) (eV/Å)
- `max_force` (float): 最大力分量的绝对值 (eV/Å)
- `rms_force` (float): 力的均方根 (eV/Å)
- `stress` (ndarray): 应力张量（Voigt 记号，6 分量） (eV/Å³)
- `pressure_GPa` (float): 压力 (GPa)

**示例**:

```python
# 从文件读取并计算
result = calc.single_point("MOF-5.cif")
print(f"能量: {result['energy']:.4f} eV")
print(f"最大力: {result['max_force']:.4f} eV/Å")
print(f"压力: {result['pressure_GPa']:.2f} GPa")

# 使用 ASE Atoms 对象
from ase.build import bulk
atoms = bulk('Cu', 'fcc', a=3.6)
result = calc.single_point(atoms)

# 只计算能量
result = calc.single_point("structure.cif", properties=["energy"])
```

---

#### 方法 2: optimize()

结构优化（原子位置和/或晶胞参数）。

```python
calc.optimize(
    structure: Union[str, Path, Atoms],
    fmax: float = 0.05,
    max_steps: int = 500,
    optimize_cell: bool = False,
    optimizer: str = "LBFGS",
    output_file: Optional[str] = None
) -> Dict[str, Any]
```

**参数**:
- `structure` (Union[str, Path, Atoms]): 输入结构
- `fmax` (float): 力收敛阈值 (eV/Å)，默认 0.05
  - 当所有原子的力小于 fmax 时，优化收敛
- `max_steps` (int): 最大优化步数，默认 500
- `optimize_cell` (bool): 是否优化晶胞参数，默认 False
  - `True` - 同时优化原子位置和晶胞
  - `False` - 仅优化原子位置
- `optimizer` (str): 优化算法
  - `"LBFGS"` - Limited-memory BFGS（默认，推荐）
  - `"BFGS"` - BFGS 算法
  - `"FIRE"` - Fast Inertial Relaxation Engine
- `output_file` (Optional[str]): 输出结构文件路径

**返回值**: Dict[str, Any]
- `converged` (bool): 是否收敛
- `steps` (int): 实际优化步数
- `final_energy` (float): 最终能量 (eV)
- `atoms` (Atoms): 优化后的 ASE Atoms 对象

**示例**:

```python
# 基本优化（仅位置）
result = calc.optimize("MOF.cif", fmax=0.01)
if result['converged']:
    print(f"优化收敛，用了 {result['steps']} 步")
    print(f"最终能量: {result['final_energy']:.6f} eV")

# 优化位置和晶胞
result = calc.optimize(
    "MOF.cif",
    fmax=0.01,
    optimize_cell=True,
    output_file="MOF_optimized.cif"
)

# 使用 FIRE 优化器
result = calc.optimize(
    atoms,
    fmax=0.02,
    optimizer="FIRE",
    max_steps=1000
)

# 访问优化后的结构
optimized_atoms = result['atoms']
optimized_atoms.write("optimized.xyz")
```

---

#### 方法 3: run_md()

分子动力学模拟。

```python
calc.run_md(
    structure: Union[str, Path, Atoms],
    ensemble: str = "nvt",
    temperature: float = 300.0,
    temperature_K: Optional[float] = None,
    pressure: Optional[float] = None,
    pressure_GPa: Optional[float] = None,
    timestep: float = 1.0,
    steps: int = 10000,
    trajectory_file: Optional[str] = None,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100
) -> Atoms
```

**参数**:
- `structure` (Union[str, Path, Atoms]): 初始结构
- `ensemble` (str): 系综类型
  - `"nve"` - 微正则系综（恒定 N, V, E）
  - `"nvt"` - 正则系综（恒定 N, V, T）
  - `"npt"` - 等温等压系综（恒定 N, P, T）
- `temperature` (float): 温度 (K)，默认 300.0
- `temperature_K` (Optional[float]): 温度 (K)（temperature 的别名）
- `pressure` (Optional[float]): 压力 (GPa)（仅用于 NPT）
- `pressure_GPa` (Optional[float]): 压力 (GPa)（pressure 的别名）
- `timestep` (float): 时间步长 (fs)，默认 1.0
- `steps` (int): MD 步数，默认 10000
- `trajectory_file` (Optional[str]): 轨迹输出文件路径
- `trajectory` (Optional[str]): 轨迹文件路径（trajectory_file 的别名）
- `logfile` (Optional[str]): MD 日志文件路径
- `log_interval` (int): 日志记录间隔（步数），默认 100

**返回值**: Atoms - MD 后的最终结构

**示例**:

```python
# NVT 模拟
final_atoms = calc.run_md(
    "MOF.cif",
    ensemble="nvt",
    temperature_K=300,
    timestep=1.0,
    steps=50000,
    trajectory="md_trajectory.traj",
    logfile="md.log"
)

# NPT 模拟
final_atoms = calc.run_md(
    atoms,
    ensemble="npt",
    temperature=500,
    pressure_GPa=1.0,
    timestep=1.5,
    steps=100000
)

# NVE 模拟（微正则系综）
final_atoms = calc.run_md(
    "structure.cif",
    ensemble="nve",
    timestep=0.5,
    steps=20000,
    trajectory="nve_traj.traj"
)

# 读取和分析轨迹
from ase.io import read
trajectory = read("md_trajectory.traj", ":")
print(f"轨迹包含 {len(trajectory)} 帧")
```

---

#### 方法 4: calculate_phonon()

声子和热力学性质计算。

```python
calc.calculate_phonon(
    structure: Union[str, Path, Atoms],
    supercell: List[int] = [2, 2, 2],
    mesh: List[int] = [20, 20, 20],
    temperature_range: tuple = (0, 500, 50)
) -> Dict[str, Any]
```

**参数**:
- `structure` (Union[str, Path, Atoms]): 原胞结构
- `supercell` (List[int]): 超胞大小 [nx, ny, nz]，默认 [2, 2, 2]
  - 更大的超胞提供更准确的声子谱，但计算成本更高
- `mesh` (List[int]): q 点网格 [mx, my, mz]，默认 [20, 20, 20]
  - 用于计算声子态密度
- `temperature_range` (tuple): 温度范围 (T_min, T_max, T_step) (K)
  - 默认 (0, 500, 50)，即 0 到 500 K，间隔 50 K

**返回值**: Dict[str, Any]
- `ZPE` (float): 零点能 (eV)
- `band_structure` (ndarray): 声子能带结构
- `DOS` (dict): 声子态密度
  - `frequencies` (ndarray): 频率 (THz)
  - `dos` (ndarray): 态密度
- `thermal_properties` (dict): 热力学性质随温度变化
  - `temperatures` (ndarray): 温度点 (K)
  - `free_energy` (ndarray): 自由能 (eV)
  - `entropy` (ndarray): 熵 (eV/K)
  - `heat_capacity` (ndarray): 热容 (eV/K)

**示例**:

```python
# 基本声子计算
result = calc.calculate_phonon("MOF.cif")
print(f"零点能: {result['ZPE']:.4f} eV")

# 使用更大超胞和更密的 q 点网格
result = calc.calculate_phonon(
    "structure.cif",
    supercell=[3, 3, 3],
    mesh=[30, 30, 30]
)

# 自定义温度范围
result = calc.calculate_phonon(
    atoms,
    temperature_range=(0, 1000, 100)
)

# 访问热力学性质
temps = result['thermal_properties']['temperatures']
Cv = result['thermal_properties']['heat_capacity']
for T, cv in zip(temps, Cv):
    print(f"T = {T:.0f} K: Cv = {cv:.6f} eV/K")

# 绘制声子态密度
import matplotlib.pyplot as plt
dos_data = result['DOS']
plt.plot(dos_data['frequencies'], dos_data['dos'])
plt.xlabel('频率 (THz)')
plt.ylabel('态密度')
plt.savefig('phonon_dos.png')
```

---

#### 方法 5: calculate_bulk_modulus()

体弹性模量计算（使用状态方程）。

```python
calc.calculate_bulk_modulus(
    structure: Union[str, Path, Atoms],
    strain_range: float = 0.05,
    npoints: int = 11
) -> Dict[str, float]
```

**参数**:
- `structure` (Union[str, Path, Atoms]): 输入结构
- `strain_range` (float): 应变范围，默认 0.05（±5%）
  - 体积将在 (1-strain_range) 到 (1+strain_range) 范围内变化
- `npoints` (int): 应变点数量，默认 11
  - 更多点可以提高拟合精度

**返回值**: Dict[str, float]
- `bulk_modulus` (float): 体弹性模量 (GPa)
- `bulk_modulus_derivative` (float): 体弹性模量的压力导数
- `E0` (float): 平衡能量 (eV)
- `V0` (float): 平衡体积 (Å³)
- `volumes` (list): 体积点 (Å³)
- `energies` (list): 对应能量 (eV)
- `eos_name` (str): 状态方程名称（如 "Birch-Murnaghan"）

**示例**:

```python
# 基本体弹性模量计算
result = calc.calculate_bulk_modulus("MOF.cif")
print(f"体弹性模量: {result['bulk_modulus']:.2f} GPa")
print(f"平衡体积: {result['V0']:.2f} Å³")

# 使用更大应变范围和更多点
result = calc.calculate_bulk_modulus(
    atoms,
    strain_range=0.10,  # ±10%
    npoints=21
)

# 绘制 E-V 曲线
import matplotlib.pyplot as plt
plt.scatter(result['volumes'], result['energies'], label='计算点')
plt.xlabel('体积 (Å³)')
plt.ylabel('能量 (eV)')
plt.legend()
plt.savefig('EV_curve.png')

# 体弹性模量的导数
B = result['bulk_modulus']
B_prime = result['bulk_modulus_derivative']
print(f"B = {B:.2f} GPa")
print(f"B' = {B_prime:.2f}")
```

---

#### 方法 6: get_calculator()

返回底层的 ASE 计算器对象，用于直接使用。

```python
calc.get_calculator() -> Calculator
```

**参数**: 无

**返回值**: Calculator - ASE 计算器对象

**示例**:

```python
# 获取计算器并附加到 Atoms 对象
calculator = calc.get_calculator()
atoms.calc = calculator

# 直接使用 ASE 方法
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

# 用于 ASE 的优化器
from ase.optimize import BFGS
atoms.calc = calc.get_calculator()
opt = BFGS(atoms)
opt.run(fmax=0.01)
```

---

## 工具模块 (Utils)

`equiformerv2_inference.utils` 模块提供了设备管理和输入输出功能。

### 设备管理

#### get_device()

获取适合计算的设备。

```python
from equiformerv2_inference.utils import get_device

get_device(device: str = "auto") -> str
```

**参数**:
- `device` (str): 设备规格
  - `"auto"` - 自动检测（优先 GPU）
  - `"cpu"` - 强制使用 CPU
  - `"cuda"` - 强制使用 CUDA

**返回值**: str - 设备字符串（"cpu" 或 "cuda"）

**异常**:
- `ValueError`: 当请求 CUDA 但不可用时

**示例**:

```python
device = get_device("auto")  # 自动选择
device = get_device("cuda")  # 强制 GPU
device = get_device("cpu")   # 强制 CPU
```

---

#### get_available_devices()

获取所有可用设备列表。

```python
from equiformerv2_inference.utils import get_available_devices

get_available_devices() -> List[str]
```

**参数**: 无

**返回值**: List[str] - 可用设备列表，如 ["cpu", "cuda:0", "cuda:1"]

**示例**:

```python
devices = get_available_devices()
print(f"可用设备: {devices}")

# 示例输出:
# 单 GPU 系统: ["cpu", "cuda:0"]
# 多 GPU 系统: ["cpu", "cuda:0", "cuda:1", "cuda:2"]
# 无 GPU 系统: ["cpu"]
```

---

#### print_device_info()

打印详细的设备信息。

```python
from equiformerv2_inference.utils import print_device_info

print_device_info() -> None
```

**参数**: 无

**返回值**: None（打印输出到控制台）

**示例**:

```python
print_device_info()

# 示例输出:
# ==================================================
# Device Information
# ==================================================
# PyTorch version: 2.0.1
# CUDA available: True
# CUDA version: 11.8
# Number of GPUs: 2
#
# GPU 0:
#   Name: NVIDIA A100-SXM4-40GB
#   Memory: 40.00 GB
#   Compute Capability: 8.0
#
# GPU 1:
#   Name: NVIDIA A100-SXM4-40GB
#   Memory: 40.00 GB
#   Compute Capability: 8.0
# ==================================================
```

---

### 输入输出

#### read_structure()

从文件加载原子结构。

```python
from equiformerv2_inference.utils import read_structure

read_structure(
    filepath: Union[str, Path],
    index: int = -1
) -> Atoms
```

**参数**:
- `filepath` (Union[str, Path]): 结构文件路径
  - 支持格式: CIF, POSCAR, VASP, XYZ, PDB, Quantum Espresso, etc.
- `index` (int): 轨迹文件的帧索引，默认 -1（最后一帧）

**返回值**: Atoms - ASE Atoms 对象

**异常**:
- `FileNotFoundError`: 文件不存在

**示例**:

```python
# 读取 CIF 文件
atoms = read_structure("MOF-5.cif")

# 读取 POSCAR
atoms = read_structure("POSCAR")

# 读取轨迹的第一帧
atoms = read_structure("trajectory.traj", index=0)

# 读取 XYZ 文件的所有帧（使用 ":"）
from ase.io import read
all_frames = read("trajectory.xyz", ":")
```

---

#### write_structure()

保存原子结构到文件。

```python
from equiformerv2_inference.utils import write_structure

write_structure(
    atoms: Atoms,
    filepath: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> None
```

**参数**:
- `atoms` (Atoms): ASE Atoms 对象
- `filepath` (Union[str, Path]): 输出文件路径
- `format` (Optional[str]): 文件格式（如果为 None，从扩展名自动检测）
- `**kwargs`: 传递给 `ase.io.write` 的其他参数

**返回值**: None

**示例**:

```python
# 自动检测格式（从扩展名）
write_structure(atoms, "output.cif")
write_structure(atoms, "output.xyz")

# 显式指定格式
write_structure(atoms, "output.dat", format="xyz")

# 保存到轨迹文件（追加模式）
from ase.io import Trajectory
traj = Trajectory("traj.traj", "a")
traj.write(atoms)
traj.close()
```

---

#### validate_structure()

验证原子结构的有效性。

```python
from equiformerv2_inference.utils import validate_structure

validate_structure(atoms: Atoms) -> None
```

**参数**:
- `atoms` (Atoms): 要验证的 ASE Atoms 对象

**返回值**: None

**异常**:
- `ValueError`: 结构无效时抛出，包含错误描述

**示例**:

```python
try:
    validate_structure(atoms)
    print("结构有效")
except ValueError as e:
    print(f"结构验证失败: {e}")

# 验证检查包括:
# - 结构是否包含原子
# - 是否定义了晶胞或周期性边界条件
# - 原子间距是否过近（< 0.5 Å）
```

---

#### get_structure_info()

获取原子结构的基本信息。

```python
from equiformerv2_inference.utils import get_structure_info

get_structure_info(atoms: Atoms) -> dict
```

**参数**:
- `atoms` (Atoms): ASE Atoms 对象

**返回值**: dict - 包含结构信息的字典
- `formula` (str): 化学式
- `n_atoms` (int): 原子数
- `volume` (float): 体积 (Å³)
- `cell_parameters` (ndarray): 晶胞参数 [a, b, c, α, β, γ]
- `pbc` (ndarray): 周期性边界条件 [x, y, z]
- `elements` (list): 元素列表

**示例**:

```python
info = get_structure_info(atoms)
print(f"化学式: {info['formula']}")
print(f"原子数: {info['n_atoms']}")
print(f"体积: {info['volume']:.2f} Å³")
print(f"晶胞参数: {info['cell_parameters']}")
```

---

## 任务模块 (Tasks)

`equiformerv2_inference.tasks` 模块提供了各种计算任务的底层实现。

### 静态计算

#### calculate_single_point()

执行单点能量和力计算。

```python
from equiformerv2_inference.tasks import calculate_single_point

calculate_single_point(
    atoms: Atoms,
    calculator,
    properties: Optional[List[str]] = None
) -> Dict[str, Any]
```

**参数**:
- `atoms` (Atoms): ASE Atoms 对象
- `calculator`: ASE 计算器对象
- `properties` (Optional[List[str]]): 要计算的性质列表
  - 默认: ["energy", "forces", "stress"]

**返回值**: Dict[str, Any] - 计算结果字典（同 `EquiformerV2Inference.single_point()`）

**示例**:

```python
from equiformerv2_inference import EquiformerV2Inference
from equiformerv2_inference.tasks import calculate_single_point
from ase.build import bulk

calc = EquiformerV2Inference()
atoms = bulk('Cu', 'fcc', a=3.6)

# 直接使用任务函数
result = calculate_single_point(atoms, calc.get_calculator())
print(f"能量: {result['energy']:.6f} eV")
```

---

### 分子动力学

#### run_md()

执行分子动力学模拟。

```python
from equiformerv2_inference.tasks import run_md

run_md(
    atoms: Atoms,
    calculator,
    ensemble: str = "nvt",
    temperature: float = 300.0,
    pressure: Optional[float] = None,
    timestep: float = 1.0,
    steps: int = 10000,
    trajectory_file: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100
) -> Atoms
```

**参数**: （同 `EquiformerV2Inference.run_md()`）

**返回值**: Atoms - 最终结构

**示例**:

```python
from equiformerv2_inference.tasks import run_md

atoms.calc = calc.get_calculator()
final_atoms = run_md(
    atoms=atoms,
    calculator=atoms.calc,
    ensemble="nvt",
    temperature=500,
    steps=50000,
    trajectory_file="md.traj"
)
```

---

### 声子计算

#### calculate_phonon()

计算声子和热力学性质。

```python
from equiformerv2_inference.tasks import calculate_phonon

calculate_phonon(
    atoms: Atoms,
    calculator,
    supercell: List[int] = [2, 2, 2],
    mesh: List[int] = [20, 20, 20],
    temperature_range: tuple = (0, 500, 50)
) -> Dict[str, Any]
```

**参数**: （同 `EquiformerV2Inference.calculate_phonon()`）

**返回值**: Dict[str, Any] - 声子计算结果

**示例**:

```python
from equiformerv2_inference.tasks import calculate_phonon

atoms.calc = calc.get_calculator()
result = calculate_phonon(
    atoms=atoms,
    calculator=atoms.calc,
    supercell=[3, 3, 3],
    mesh=[30, 30, 30]
)
```

---

### 力学性质

#### calculate_bulk_modulus()

计算体弹性模量。

```python
from equiformerv2_inference.tasks import calculate_bulk_modulus

calculate_bulk_modulus(
    atoms: Atoms,
    calculator,
    strain_range: float = 0.05,
    npoints: int = 11
) -> Dict[str, float]
```

**参数**: （同 `EquiformerV2Inference.calculate_bulk_modulus()`）

**返回值**: Dict[str, float] - 体弹性模量和拟合参数

**示例**:

```python
from equiformerv2_inference.tasks import calculate_bulk_modulus

atoms.calc = calc.get_calculator()
result = calculate_bulk_modulus(
    atoms=atoms,
    calculator=atoms.calc,
    strain_range=0.08,
    npoints=17
)
print(f"B = {result['bulk_modulus']:.2f} GPa")
```

---

## 命令行接口 (CLI)

EquiformerV2 Inference 提供了 `equiformerv2-infer` 命令行工具。

### 通用选项

所有子命令共享以下选项：

- `--model MODEL`: 模型名称或路径
  - 默认: `equiformer_v2`（即 EquiformerV2-31M-S2EF）
- `--device DEVICE`: 计算设备
  - 可选值: `auto`, `cpu`, `cuda`
  - 默认: `auto`

---

### 子命令 1: single-point

单点能量和力计算。

```bash
equiformerv2-infer single-point STRUCTURE [OPTIONS]
```

**位置参数**:
- `STRUCTURE`: 结构文件路径

**可选参数**:
- `--output FILE`: 输出结果文件（JSON 格式）

**示例**:

```bash
# 基本单点计算
equiformerv2-infer single-point MOF-5.cif

# 保存结果到 JSON
equiformerv2-infer single-point structure.cif --output result.json

# 使用 CPU
equiformerv2-infer single-point MOF.cif --device cpu

# 使用特定模型
equiformerv2-infer single-point structure.cif --model EquiformerV2-153M-S2EF
```

**输出格式（JSON）**:
```json
{
  "energy": -123.456789,
  "energy_per_atom": -1.234567,
  "forces": [[0.001, -0.002, 0.003], ...],
  "max_force": 0.05,
  "rms_force": 0.02,
  "stress": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
  "pressure_GPa": 1.5
}
```

---

### 子命令 2: optimize

结构优化。

```bash
equiformerv2-infer optimize STRUCTURE [OPTIONS]
```

**位置参数**:
- `STRUCTURE`: 输入结构文件路径

**可选参数**:
- `--fmax FLOAT`: 力收敛阈值 (eV/Å)，默认 0.05
- `--steps INT`: 最大优化步数，默认 500
- `--optimizer NAME`: 优化器名称
  - 可选值: `LBFGS`, `BFGS`, `FIRE`
  - 默认: `LBFGS`
- `--cell`: 是否优化晶胞参数（标志）
- `--output FILE`: 输出优化后的结构文件
- `--trajectory FILE`: 保存优化轨迹

**示例**:

```bash
# 基本优化
equiformerv2-infer optimize MOF.cif --fmax 0.01

# 优化位置和晶胞
equiformerv2-infer optimize MOF.cif --fmax 0.01 --cell --output optimized.cif

# 使用 FIRE 优化器
equiformerv2-infer optimize structure.cif --optimizer FIRE --steps 1000

# 保存轨迹
equiformerv2-infer optimize MOF.cif --trajectory opt_trajectory.traj
```

---

### 子命令 3: md

分子动力学模拟。

```bash
equiformerv2-infer md STRUCTURE [OPTIONS]
```

**位置参数**:
- `STRUCTURE`: 初始结构文件路径

**可选参数**:
- `--ensemble NAME`: MD 系综
  - 可选值: `nve`, `nvt`, `npt`
  - 默认: `nvt`
- `--temp FLOAT`: 温度 (K)，默认 300
- `--pressure FLOAT`: 压力 (GPa)，仅用于 NPT
- `--steps INT`: MD 步数，默认 1000
- `--timestep FLOAT`: 时间步长 (fs)，默认 1.0
- `--trajectory FILE`: 轨迹输出文件
- `--logfile FILE`: 日志文件

**示例**:

```bash
# NVT 模拟
equiformerv2-infer md MOF.cif --ensemble nvt --temp 300 --steps 50000

# NPT 模拟
equiformerv2-infer md structure.cif --ensemble npt --temp 500 --pressure 1.0 --steps 100000

# 保存轨迹和日志
equiformerv2-infer md MOF.cif --steps 10000 --trajectory md_traj.traj --logfile md.log

# NVE 模拟
equiformerv2-infer md structure.cif --ensemble nve --timestep 0.5 --steps 20000
```

---

### 子命令 4: phonon

声子计算。

```bash
equiformerv2-infer phonon STRUCTURE [OPTIONS]
```

**位置参数**:
- `STRUCTURE`: 原胞结构文件路径

**可选参数**:
- `--supercell NX NY NZ`: 超胞大小，默认 `2 2 2`
- `--mesh MX MY MZ`: q 点网格，默认 `20 20 20`
- `--temp-range MIN MAX STEP`: 温度范围 (K)
- `--output-dir DIR`: 输出目录

**示例**:

```bash
# 基本声子计算
equiformerv2-infer phonon MOF.cif

# 使用更大超胞
equiformerv2-infer phonon structure.cif --supercell 3 3 3 --mesh 30 30 30

# 自定义温度范围
equiformerv2-infer phonon MOF.cif --temp-range 0 1000 100

# 指定输出目录
equiformerv2-infer phonon structure.cif --output-dir phonon_results/
```

---

### 子命令 5: bulk-modulus

体弹性模量计算。

```bash
equiformerv2-infer bulk-modulus STRUCTURE [OPTIONS]
```

**位置参数**:
- `STRUCTURE`: 结构文件路径

**可选参数**:
- `--points INT`: 体积点数量，默认 11
- `--strain-range FLOAT`: 应变范围 (±)，默认 0.05
- `--eos NAME`: 状态方程类型
  - 可选值: `birchmurnaghan`, `murnaghan`, `vinet`
  - 默认: `birchmurnaghan`

**示例**:

```bash
# 基本体弹性模量计算
equiformerv2-infer bulk-modulus MOF.cif

# 使用更多点和更大应变
equiformerv2-infer bulk-modulus structure.cif --points 21 --strain-range 0.10

# 使用不同的状态方程
equiformerv2-infer bulk-modulus MOF.cif --eos murnaghan
```

---

### 帮助信息

```bash
# 查看主帮助
equiformerv2-infer --help

# 查看子命令帮助
equiformerv2-infer single-point --help
equiformerv2-infer optimize --help
equiformerv2-infer md --help
equiformerv2-infer phonon --help
equiformerv2-infer bulk-modulus --help
```

---

## 完整使用示例

### 示例 1: MOF 材料的完整工作流

```python
from equiformerv2_inference import EquiformerV2Inference

# 1. 初始化
calc = EquiformerV2Inference(
    model_name="EquiformerV2-31M-S2EF",
    device="cuda"
)

# 2. 单点计算
result = calc.single_point("MOF-5.cif")
print(f"初始能量: {result['energy']:.4f} eV")
print(f"最大力: {result['max_force']:.4f} eV/Å")

# 3. 结构优化
opt_result = calc.optimize(
    "MOF-5.cif",
    fmax=0.01,
    optimize_cell=True,
    output_file="MOF-5_optimized.cif"
)
print(f"优化收敛: {opt_result['converged']}")
print(f"最终能量: {opt_result['final_energy']:.4f} eV")

# 4. 分子动力学
final_atoms = calc.run_md(
    opt_result['atoms'],
    ensemble="nvt",
    temperature_K=300,
    steps=50000,
    trajectory="MOF-5_md.traj"
)

# 5. 声子计算
phonon_result = calc.calculate_phonon(
    "MOF-5_optimized.cif",
    supercell=[2, 2, 2]
)
print(f"零点能: {phonon_result['ZPE']:.4f} eV")

# 6. 体弹性模量
bulk_result = calc.calculate_bulk_modulus("MOF-5_optimized.cif")
print(f"体弹性模量: {bulk_result['bulk_modulus']:.2f} GPa")
```

### 示例 2: 批量计算

```python
from equiformerv2_inference import EquiformerV2Inference
from pathlib import Path
import json

calc = EquiformerV2Inference(device="cuda")

# 批量处理多个结构
structures = Path("structures/").glob("*.cif")
results = {}

for structure_file in structures:
    name = structure_file.stem
    result = calc.single_point(structure_file)
    results[name] = {
        "energy": result['energy'],
        "max_force": result['max_force'],
        "pressure_GPa": result['pressure_GPa']
    }

# 保存结果
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### 示例 3: 高通量筛选

```python
from equiformerv2_inference import EquiformerV2Inference
from ase.io import read
import pandas as pd

calc = EquiformerV2Inference(model_name="EquiformerV2-153M-S2EF")

# 读取候选结构数据库
structures = read("candidates.traj", ":")
results = []

for i, atoms in enumerate(structures):
    # 优化
    opt_result = calc.optimize(atoms, fmax=0.02)
    
    # 计算性质
    bulk_mod = calc.calculate_bulk_modulus(opt_result['atoms'])
    
    results.append({
        "structure_id": i,
        "energy_per_atom": opt_result['final_energy'] / len(atoms),
        "bulk_modulus_GPa": bulk_mod['bulk_modulus'],
        "converged": opt_result['converged']
    })

# 保存为 CSV
df = pd.DataFrame(results)
df.to_csv("screening_results.csv", index=False)
```

---

## 常见问题 (FAQ)

### Q1: 如何选择模型？

- **EquiformerV2-31M-S2EF**: 默认选择，速度快，精度良好
- **EquiformerV2-153M-S2EF**: 更高精度，但计算成本更高

### Q2: 如何选择设备？

- `device="auto"`: 推荐，自动选择最佳设备
- `device="cuda"`: 强制使用 GPU（需要 GPU 可用）
- `device="cpu"`: 强制使用 CPU（较慢但无需 GPU）

### Q3: 优化不收敛怎么办？

1. 增加 `max_steps`
2. 放宽 `fmax` 阈值
3. 尝试不同的 `optimizer`（FIRE 对难收敛的系统可能更好）
4. 检查初始结构是否合理

### Q4: MD 模拟的时间步长如何选择？

- 一般系统: 1.0 fs
- 轻原子（H）: 0.5 fs
- 重原子: 1.5-2.0 fs

### Q5: 声子计算中超胞大小如何选择？

- 小分子/简单晶体: [2, 2, 2]
- 复杂材料: [3, 3, 3] 或更大
- 原则: 超胞越大，精度越高，但计算成本呈立方增长

---

## 性能优化建议

1. **使用 GPU**: 对于大系统（>100 原子），GPU 可提速 10-100 倍
2. **批量计算**: 使用脚本自动化批量任务
3. **合理的收敛阈值**: 不要过度追求极高精度
4. **轨迹文件**: 大规模 MD 时定期保存轨迹，避免内存溢出

---

## 许可证

MIT License

---

## 引用

如果使用 EquiformerV2 Inference，请引用：

```bibtex
@article{equiformerv2,
  title={EquiformerV2: Improved Equivariant Transformer for Molecular Modeling},
  author={...},
  journal={...},
  year={2024}
}
```

---

**文档版本**: 0.1.0  
**最后更新**: 2026年1月7日
