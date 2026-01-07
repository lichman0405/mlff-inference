# GRACE Inference API 参考文档

本文档详细说明 GRACE (GRAph Convolutional E3-equivariant neural network) 推理库的所有接口、类、方法和模块。

**版本要求**：
- Python >= 3.9
- ASE >= 3.22.0
- Phonopy >= 2.20.0
- PyTorch >= 2.0.0
- DGL >= 1.0.0

---

## 目录

1. [核心类](#1-核心类)
   - [GRACEInference](#11-graceinference-类)
2. [任务模块](#2-任务模块)
   - [静态计算](#21-静态计算模块-static)
   - [结构优化](#22-结构优化模块-static)
   - [分子动力学](#23-分子动力学模块-dynamics)
   - [声子计算](#24-声子计算模块-phonon)
   - [力学性质](#25-力学性质模块-mechanics)
   - [吸附能计算](#26-吸附能计算模块-adsorption)
3. [工具模块](#3-工具模块)
   - [设备管理](#31-设备管理-device)
   - [输入输出](#32-输入输出-io)
4. [CLI 命令行工具](#4-cli-命令行工具)
5. [数据结构](#5-数据结构)
6. [异常处理](#6-异常处理)

---

## 1. 核心类

### 1.1 GRACEInference 类

**完整路径**: `grace_inference.core.GRACEInference`

GRACE 推理计算器的统一接口，提供图注意力机制的机器学习力场计算功能。

#### 1.1.1 类初始化

```python
GRACEInference(
    model_name: str = "grace-2l",
    model_path: Optional[str] = None,
    device: str = "auto",
    dtype: str = "float32"
)
```

**参数**：
- `model_name` (str): 模型名称
  - `"grace-2l"`: 2层图卷积网络（默认，适合快速计算）
  - `"grace-3l"`: 3层图卷积网络（更高精度）
  - 默认值: `"grace-2l"`
- `model_path` (Optional[str]): 自定义模型检查点路径，如果提供则覆盖 `model_name`
- `device` (str): 计算设备
  - `"auto"`: 自动检测（优先使用 CUDA）
  - `"cpu"`: 强制使用 CPU
  - `"cuda"`: 强制使用 CUDA
  - `"cuda:0"`, `"cuda:1"`: 指定 GPU 设备
  - 默认值: `"auto"`
- `dtype` (str): 数据类型，`"float32"` 或 `"float64"`，默认 `"float32"`

**返回值**：GRACEInference 实例

**异常**：
- `ImportError`: 如果 GRACE 计算器未安装
- `ValueError`: 如果设备参数无效或 CUDA 不可用

**示例**：
```python
from grace_inference import GRACEInference

# 基本用法：使用默认模型和自动设备检测
calc = GRACEInference()

# 指定模型和设备
calc = GRACEInference(model_name="grace-3l", device="cuda")

# 使用自定义模型
calc = GRACEInference(model_path="/path/to/custom_model.pt", device="cuda:0")
```

---

#### 1.1.2 单点能量计算

```python
GRACEInference.single_point(
    structure: Union[str, Path, Atoms],
    properties: List[str] = ["energy", "forces", "stress"]
) -> Dict[str, Any]
```

**物理意义**：
计算给定几何构型的总能量、原子受力和应力张量。这是所有量子化学和分子模拟的基础，可用于：
- 评估结构稳定性（能量越低越稳定）
- 比较不同构型的相对稳定性
- 计算反应能、吸附能等能量差
- 判断结构是否处于平衡态（力接近零）

**参数**：
- `structure` (Union[str, Path, Atoms]): 
  - 字符串/Path: 结构文件路径（支持 CIF, POSCAR, XYZ 等）
  - Atoms: ASE Atoms 对象
- `properties` (List[str]): 要计算的性质列表
  - 可选值: `"energy"`, `"forces"`, `"stress"`
  - 默认: `["energy", "forces", "stress"]`

**返回值**：Dict[str, Any]，包含：
- `energy` (float): 总能量，单位 eV
- `energy_per_atom` (float): 每原子能量，单位 eV/atom
- `forces` (np.ndarray): 原子受力，形状 (N, 3)，单位 eV/Å
- `max_force` (float): 最大力分量，单位 eV/Å
- `rms_force` (float): 均方根力，单位 eV/Å
- `stress` (np.ndarray): Voigt 应力张量，形状 (6,)，单位 eV/Å³
- `pressure_GPa` (float): 压强，单位 GPa

**示例**：
```python
from grace_inference import GRACEInference

calc = GRACEInference(device="cuda")

# 从文件读取
result = calc.single_point("MOF-5.cif")
print(f"Energy: {result['energy']:.6f} eV")
print(f"Max Force: {result['max_force']:.6f} eV/Å")
print(f"Pressure: {result['pressure_GPa']:.2f} GPa")

# 只计算能量
result = calc.single_point("structure.cif", properties=["energy"])
```

---

#### 1.1.3 结构优化

```python
GRACEInference.optimize(
    structure: Union[str, Path, Atoms],
    fmax: float = 0.05,
    steps: int = 500,
    optimizer: str = "LBFGS",
    optimize_cell: bool = False,
    trajectory: Optional[str] = None,
    output: Optional[str] = None
) -> Atoms
```

**物理意义**：
优化原子结构，使其达到能量最小化的平衡状态。通过迭代调整原子位置（和晶胞参数），直到所有原子受力小于收敛标准。用于：
- 获得稳定结构
- 消除结构中的应力
- 作为分子动力学的初始结构
- 研究材料的基态性质

**参数**：
- `structure` (Union[str, Path, Atoms]): 输入结构
- `fmax` (float): 力收敛标准，单位 eV/Å
  - 当所有原子受力 < fmax 时优化收敛
  - 推荐值：粗优化 0.05，精细优化 0.01
- `steps` (int): 最大优化步数，默认 500
- `optimizer` (str): 优化算法
  - `"LBFGS"`: Limited-memory BFGS（默认，适合大多数情况）
  - `"BFGS"`: Broyden-Fletcher-Goldfarb-Shanno
  - `"FIRE"`: Fast Inertial Relaxation Engine（适合远离平衡态的结构）
- `optimize_cell` (bool): 是否同时优化晶胞参数，默认 False
- `trajectory` (Optional[str]): 优化轨迹文件保存路径
- `output` (Optional[str]): 优化后结构保存路径

**返回值**：Atoms，优化后的结构

**示例**：
```python
# 基本优化：只优化原子位置
optimized = calc.optimize("input.cif", fmax=0.01)
optimized.write("optimized.cif")

# 同时优化晶胞参数
optimized = calc.optimize(
    "input.cif",
    fmax=0.01,
    optimize_cell=True,
    trajectory="opt.traj",
    output="optimized.cif"
)

# 使用 FIRE 算法优化
optimized = calc.optimize("input.cif", optimizer="FIRE", steps=1000)
```

---

#### 1.1.4 分子动力学模拟

```python
GRACEInference.molecular_dynamics(
    structure: Union[str, Path, Atoms],
    ensemble: str = "nvt",
    temperature_K: float = 300,
    pressure_GPa: Optional[float] = None,
    timestep: float = 1.0,
    steps: int = 1000,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100
) -> Atoms
```

**物理意义**：
运行分子动力学模拟，模拟原子在有限温度下的热运动。通过求解牛顿运动方程，研究材料的动力学行为和热力学性质。用于：
- 研究热力学性质（热膨胀系数、比热容等）
- 模拟扩散过程
- 研究相变和结构演化
- 计算时间平均性质

**参数**：
- `structure` (Union[str, Path, Atoms]): 初始结构
- `ensemble` (str): MD 系综
  - `"nve"`: 微正则系综（恒定 N, V, E）
  - `"nvt"`: 正则系综（恒定 N, V, T，默认）
  - `"npt"`: 等温等压系综（恒定 N, P, T）
- `temperature_K` (float): 目标温度，单位 K，默认 300
- `pressure_GPa` (Optional[float]): NPT 系综的目标压强，单位 GPa
- `timestep` (float): 时间步长，单位 fs，默认 1.0
- `steps` (int): MD 步数，默认 1000
- `trajectory` (Optional[str]): 轨迹文件保存路径
- `logfile` (Optional[str]): 日志文件路径
- `log_interval` (int): 日志记录间隔，默认 100 步

**返回值**：Atoms，MD 模拟后的最终结构

**示例**：
```python
# NVT 分子动力学（300K，10 ps）
final = calc.molecular_dynamics(
    "init.cif",
    ensemble="nvt",
    temperature_K=300,
    steps=10000,
    trajectory="md_nvt.traj",
    logfile="md.log"
)

# NPT 分子动力学（500K，1 GPa）
final = calc.molecular_dynamics(
    "init.cif",
    ensemble="npt",
    temperature_K=500,
    pressure_GPa=1.0,
    steps=20000,
    trajectory="md_npt.traj"
)

# NVE 分子动力学
final = calc.molecular_dynamics(
    "init.cif",
    ensemble="nve",
    steps=5000,
    trajectory="md_nve.traj"
)
```

---

#### 1.1.5 声子计算

```python
GRACEInference.phonon(
    structure: Union[str, Path, Atoms],
    supercell: tuple = (2, 2, 2),
    mesh: tuple = (20, 20, 20),
    temperature_range: Optional[tuple] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]
```

**物理意义**：
计算晶格振动（声子）性质，通过有限位移法构造力常数矩阵，求解本征值问题得到声子色散关系。用于：
- 判断结构动力学稳定性（无虚频表示稳定）
- 计算热力学性质（比热容、自由能、熵等）
- 研究相变温度
- 计算热导率

**参数**：
- `structure` (Union[str, Path, Atoms]): 输入结构（应为优化后的平衡结构）
- `supercell` (tuple): 超胞尺寸 (nx, ny, nz)
  - 更大的超胞提高精度但增加计算量
  - 推荐值：(2, 2, 2) 至 (3, 3, 3)
- `mesh` (tuple): k点网格 (nx, ny, nz)
  - 用于计算声子态密度
  - 推荐值：(20, 20, 20) 至 (50, 50, 50)
- `temperature_range` (Optional[tuple]): 温度范围 (T_min, T_max, T_step)，单位 K
  - 如果指定，计算热力学性质
  - 例如：(0, 1000, 10)
- `output_dir` (Optional[str]): 输出目录，保存声子计算结果

**返回值**：Dict[str, Any]，包含：
- `phonon` (Phonopy): Phonopy 对象
- `supercell_matrix` (list): 使用的超胞矩阵
- `displacement` (float): 位移距离
- `mesh` (tuple): k点网格
- `thermal_properties` (dict): 热力学性质（如果指定 temperature_range）
  - `temperatures` (np.ndarray): 温度数组
  - `free_energy` (np.ndarray): 自由能
  - `entropy` (np.ndarray): 熵
  - `heat_capacity` (np.ndarray): 热容

**示例**：
```python
# 基本声子计算
result = calc.phonon("structure.cif", supercell=(2, 2, 2))
phonon = result['phonon']

# 计算热力学性质
result = calc.phonon(
    "structure.cif",
    supercell=(3, 3, 3),
    mesh=(30, 30, 30),
    temperature_range=(0, 1000, 10),
    output_dir="phonon_results"
)

# 绘制声子色散
phonon = result['phonon']
phonon.auto_band_structure(plot=True).show()
```

---

#### 1.1.6 体积模量计算

```python
GRACEInference.bulk_modulus(
    structure: Union[str, Path, Atoms],
    num_points: int = 11,
    strain_range: float = 0.05,
    eos: str = "birchmurnaghan",
    plot: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]
```

**物理意义**：
通过状态方程（EOS）拟合计算体积模量，描述材料抵抗压缩的能力。通过计算不同体积下的能量，拟合 EOS 得到平衡体积、体积模量等参数。用于：
- 评估材料的可压缩性
- 研究高压下的结构行为
- 计算弹性常数
- 预测材料的力学稳定性

**参数**：
- `structure` (Union[str, Path, Atoms]): 输入结构（应为优化后的平衡结构）
- `num_points` (int): 体积采样点数，默认 11
  - 更多点提高拟合精度
- `strain_range` (float): 应变范围（±），默认 0.05（±5%）
- `eos` (str): 状态方程类型
  - `"birchmurnaghan"`: Birch-Murnaghan 方程（默认）
  - `"murnaghan"`: Murnaghan 方程
  - `"vinet"`: Vinet 方程
  - `"sjeos"`: Stabilized jellium EOS
  - `"taylor"`: Taylor 展开
- `plot` (bool): 是否生成 EOS 拟合图，默认 True
- `output_dir` (Optional[str]): 输出目录

**返回值**：Dict[str, Any]，包含：
- `bulk_modulus_GPa` (float): 体积模量，单位 GPa
- `equilibrium_volume` (float): 平衡体积，单位 Å³
- `equilibrium_energy` (float): 平衡能量，单位 eV
- `eos_parameters` (dict): EOS 拟合参数
- `volumes` (np.ndarray): 体积数组
- `energies` (np.ndarray): 能量数组

**示例**：
```python
# 基本体积模量计算
result = calc.bulk_modulus("structure.cif")
print(f"Bulk Modulus: {result['bulk_modulus_GPa']:.2f} GPa")

# 高精度计算
result = calc.bulk_modulus(
    "structure.cif",
    num_points=21,
    strain_range=0.08,
    eos="vinet",
    plot=True,
    output_dir="bulk_modulus_results"
)
```

---

#### 1.1.7 吸附能计算

```python
GRACEInference.adsorption_energy(
    host_structure: Union[str, Path, Atoms],
    adsorbate_structure: Union[str, Path, Atoms],
    combined_structure: Union[str, Path, Atoms],
    relax_host: bool = True,
    relax_adsorbate: bool = True,
    relax_combined: bool = True,
    fmax: float = 0.05
) -> Dict[str, Any]
```

**物理意义**：
计算吸附能，即吸附质分子在主体材料表面/孔道中的吸附强度。吸附能定义为：
```
E_ads = E(host+adsorbate) - E(host) - E(adsorbate)
```
负值表示吸附是放热过程（有利），用于：
- 评估气体储存性能（如 MOF 中的 CO₂、H₂ 吸附）
- 筛选催化剂材料
- 研究分子识别和分离
- 优化多孔材料设计

**参数**：
- `host_structure` (Union[str, Path, Atoms]): 主体结构（如 MOF、表面）
- `adsorbate_structure` (Union[str, Path, Atoms]): 吸附质结构（如 CO₂、H₂O）
- `combined_structure` (Union[str, Path, Atoms]): 吸附后的复合结构
- `relax_host` (bool): 是否优化主体结构，默认 True
- `relax_adsorbate` (bool): 是否优化吸附质结构，默认 True
- `relax_combined` (bool): 是否优化复合结构，默认 True
- `fmax` (float): 优化的力收敛标准，单位 eV/Å

**返回值**：Dict[str, Any]，包含：
- `adsorption_energy_eV` (float): 吸附能，单位 eV
- `adsorption_energy_kJ_mol` (float): 吸附能，单位 kJ/mol
- `host_energy` (float): 主体能量，单位 eV
- `adsorbate_energy` (float): 吸附质能量，单位 eV
- `combined_energy` (float): 复合结构能量，单位 eV
- `host_optimized` (Atoms): 优化后的主体结构
- `adsorbate_optimized` (Atoms): 优化后的吸附质结构
- `combined_optimized` (Atoms): 优化后的复合结构

**示例**：
```python
# 计算 MOF 中 CO2 的吸附能
result = calc.adsorption_energy(
    host_structure="MOF-5.cif",
    adsorbate_structure="CO2.xyz",
    combined_structure="MOF-5_CO2.cif",
    relax_all=True,
    fmax=0.01
)

print(f"Adsorption Energy: {result['adsorption_energy_eV']:.4f} eV")
print(f"Adsorption Energy: {result['adsorption_energy_kJ_mol']:.2f} kJ/mol")

# 不优化主体结构（假设已优化）
result = calc.adsorption_energy(
    host_structure="MOF-5_relaxed.cif",
    adsorbate_structure="H2.xyz",
    combined_structure="MOF-5_H2.cif",
    relax_host=False,
    relax_adsorbate=True,
    relax_combined=True
)
```

---

#### 1.1.8 工具方法

##### get_calculator()

```python
GRACEInference.get_calculator() -> Calculator
```

获取底层 ASE 计算器对象。

**返回值**：Calculator，ASE 计算器实例

**示例**：
```python
calc = GRACEInference()
ase_calc = calc.get_calculator()

# 直接使用 ASE 计算器
atoms.calc = ase_calc
energy = atoms.get_potential_energy()
```

##### set_calculator()

```python
GRACEInference.set_calculator(calculator: Calculator) -> None
```

设置自定义 ASE 计算器。

**参数**：
- `calculator` (Calculator): ASE 计算器对象

**示例**：
```python
from grace.calculator import GRACECalculator

custom_calc = GRACECalculator("/path/to/model.pt")
calc = GRACEInference()
calc.set_calculator(custom_calc)
```

---

## 2. 任务模块

任务模块（`grace_inference.tasks`）提供底层计算函数，可以直接调用而不需要 GRACEInference 类。

### 2.1 静态计算模块 (static)

**完整路径**: `grace_inference.tasks.static`

#### calculate_single_point()

```python
calculate_single_point(
    atoms: Atoms,
    calculator: Calculator,
    properties: Optional[List[str]] = None
) -> Dict[str, Any]
```

计算单点能量、力和应力。

**参数**：
- `atoms` (Atoms): ASE Atoms 对象
- `calculator` (Calculator): ASE 计算器
- `properties` (Optional[List[str]]): 要计算的性质列表

**返回值**：Dict，包含计算结果

**示例**：
```python
from ase.io import read
from grace_inference.tasks import calculate_single_point
from grace.calculator import GRACECalculator

atoms = read("structure.cif")
calc = GRACECalculator("grace-2l")
result = calculate_single_point(atoms, calc)
print(result['energy'])
```

---

### 2.2 结构优化模块 (static)

#### optimize_structure()

```python
optimize_structure(
    atoms: Atoms,
    calculator: Calculator,
    fmax: float = 0.05,
    steps: int = 500,
    optimizer: str = "LBFGS",
    optimize_cell: bool = False,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None
) -> Atoms
```

优化原子结构。

**参数**：
- `atoms` (Atoms): 输入结构
- `calculator` (Calculator): ASE 计算器
- `fmax` (float): 力收敛标准
- `steps` (int): 最大步数
- `optimizer` (str): 优化算法
- `optimize_cell` (bool): 是否优化晶胞
- `trajectory` (Optional[str]): 轨迹文件路径
- `logfile` (Optional[str]): 日志文件路径

**返回值**：Atoms，优化后的结构

**示例**：
```python
from grace_inference.tasks import optimize_structure

optimized = optimize_structure(
    atoms, 
    calc,
    fmax=0.01,
    optimize_cell=True,
    trajectory="opt.traj"
)
```

---

### 2.3 分子动力学模块 (dynamics)

**完整路径**: `grace_inference.tasks.dynamics`

#### run_md()

```python
run_md(
    atoms: Atoms,
    calculator: Calculator,
    ensemble: str = "nvt",
    temperature_K: float = 300,
    pressure_GPa: Optional[float] = None,
    timestep: float = 1.0,
    steps: int = 1000,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100
) -> Atoms
```

运行分子动力学模拟。

**参数**：
- `atoms` (Atoms): 初始结构
- `calculator` (Calculator): ASE 计算器
- `ensemble` (str): MD 系综 ("nve", "nvt", "npt")
- `temperature_K` (float): 目标温度
- `pressure_GPa` (Optional[float]): 目标压强（NPT）
- `timestep` (float): 时间步长
- `steps` (int): MD 步数
- `trajectory` (Optional[str]): 轨迹文件
- `logfile` (Optional[str]): 日志文件
- `log_interval` (int): 日志间隔

**返回值**：Atoms，最终结构

**示例**：
```python
from grace_inference.tasks import run_md

final = run_md(
    atoms,
    calc,
    ensemble="nvt",
    temperature_K=300,
    steps=10000,
    trajectory="md.traj"
)
```

---

### 2.4 声子计算模块 (phonon)

**完整路径**: `grace_inference.tasks.phonon`

#### calculate_phonon()

```python
calculate_phonon(
    atoms: Atoms,
    calculator: Calculator,
    supercell_matrix: Union[List[int], np.ndarray, int] = 2,
    displacement: float = 0.01,
    mesh: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    temperature_range: Optional[tuple] = None
) -> Dict[str, Any]
```

计算声子性质。

**参数**：
- `atoms` (Atoms): 输入结构
- `calculator` (Calculator): ASE 计算器
- `supercell_matrix` (Union[List[int], np.ndarray, int]): 超胞尺寸
- `displacement` (float): 原子位移距离，单位 Å
- `mesh` (Optional[List[int]]): k点网格
- `output_dir` (Optional[str]): 输出目录
- `temperature_range` (Optional[tuple]): 温度范围

**返回值**：Dict，包含声子计算结果

**示例**：
```python
from grace_inference.tasks import calculate_phonon

result = calculate_phonon(
    atoms,
    calc,
    supercell_matrix=[3, 3, 3],
    mesh=[30, 30, 30],
    temperature_range=(0, 1000, 10)
)
```

#### 辅助函数

##### ase_to_phonopy()

```python
ase_to_phonopy(atoms: Atoms) -> PhonopyAtoms
```

将 ASE Atoms 转换为 Phonopy Atoms。

##### phonopy_to_ase()

```python
phonopy_to_ase(phonopy_atoms: PhonopyAtoms) -> Atoms
```

将 Phonopy Atoms 转换为 ASE Atoms。

---

### 2.5 力学性质模块 (mechanics)

**完整路径**: `grace_inference.tasks.mechanics`

#### calculate_bulk_modulus()

```python
calculate_bulk_modulus(
    atoms: Atoms,
    calculator: Calculator,
    num_points: int = 11,
    strain_range: float = 0.05,
    eos: str = "birchmurnaghan",
    plot: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]
```

计算体积模量。

**参数**：
- `atoms` (Atoms): 输入结构
- `calculator` (Calculator): ASE 计算器
- `num_points` (int): 体积采样点数
- `strain_range` (float): 应变范围
- `eos` (str): 状态方程类型
- `plot` (bool): 是否生成图表
- `output_dir` (Optional[str]): 输出目录

**返回值**：Dict，包含体积模量和 EOS 参数

---

### 2.6 吸附能计算模块 (adsorption)

**完整路径**: `grace_inference.tasks.adsorption`

#### calculate_adsorption_energy()

```python
calculate_adsorption_energy(
    host: Atoms,
    adsorbate: Atoms,
    combined: Atoms,
    calculator: Calculator,
    relax_host: bool = True,
    relax_adsorbate: bool = True,
    relax_combined: bool = True,
    fmax: float = 0.05
) -> Dict[str, Any]
```

计算吸附能。

**参数**：
- `host` (Atoms): 主体结构
- `adsorbate` (Atoms): 吸附质结构
- `combined` (Atoms): 复合结构
- `calculator` (Calculator): ASE 计算器
- `relax_host` (bool): 是否优化主体
- `relax_adsorbate` (bool): 是否优化吸附质
- `relax_combined` (bool): 是否优化复合结构
- `fmax` (float): 力收敛标准

**返回值**：Dict，包含吸附能和相关性质

---

## 3. 工具模块

### 3.1 设备管理 (device)

**完整路径**: `grace_inference.utils.device`

#### get_device()

```python
get_device(device: DeviceType = "auto") -> str
```

获取计算设备。

**参数**：
- `device` (DeviceType): 设备类型
  - `"auto"`: 自动检测
  - `"cpu"`: CPU
  - `"cuda"`: CUDA
  - `"cuda:N"`: 指定 GPU

**返回值**：str，设备字符串

**异常**：
- `ValueError`: 设备不可用或参数无效

**示例**：
```python
from grace_inference.utils import get_device

device = get_device("auto")  # 自动检测
device = get_device("cuda:0")  # 使用 GPU 0
```

---

#### get_available_devices()

```python
get_available_devices() -> List[str]
```

返回所有可用设备列表。

**返回值**：List[str]，设备列表，如 `["cpu", "cuda:0", "cuda:1"]`

**示例**：
```python
from grace_inference.utils import get_available_devices

devices = get_available_devices()
print(f"Available devices: {devices}")
```

---

#### print_device_info()

```python
print_device_info() -> None
```

打印详细的设备信息。

**示例**：
```python
from grace_inference.utils import print_device_info

print_device_info()
# 输出：
# ==================================================
# Device Information
# ==================================================
# PyTorch version: 2.0.1
# CUDA available: True
# CUDA version: 11.8
# Number of GPUs: 1
# GPU 0: NVIDIA GeForce RTX 3090
# ...
```

---

### 3.2 输入输出 (io)

**完整路径**: `grace_inference.utils.io`

#### read_structure()

```python
read_structure(
    filepath: Union[str, Path],
    index: int = -1
) -> Atoms
```

从文件读取原子结构。

**参数**：
- `filepath` (Union[str, Path]): 文件路径（支持 CIF, POSCAR, XYZ, TRAJ 等）
- `index` (int): 轨迹文件的帧索引（-1 表示最后一帧）

**返回值**：Atoms，ASE Atoms 对象

**异常**：
- `FileNotFoundError`: 文件不存在

**示例**：
```python
from grace_inference.utils import read_structure

# 读取单个结构
atoms = read_structure("structure.cif")

# 读取轨迹文件的第一帧
atoms = read_structure("md.traj", index=0)
```

---

#### write_structure()

```python
write_structure(
    atoms: Atoms,
    filepath: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> None
```

保存原子结构到文件。

**参数**：
- `atoms` (Atoms): ASE Atoms 对象
- `filepath` (Union[str, Path]): 输出文件路径
- `format` (Optional[str]): 文件格式（默认从扩展名推断）
- `**kwargs`: 传递给 `ase.io.write` 的额外参数

**示例**：
```python
from grace_inference.utils import write_structure

write_structure(atoms, "output.cif")
write_structure(atoms, "output.xyz", format="xyz")
```

---

#### validate_structure()

```python
validate_structure(atoms: Atoms) -> None
```

验证原子结构的有效性。

**参数**：
- `atoms` (Atoms): 要验证的结构

**异常**：
- `ValueError`: 结构无效（如原子重叠、无晶胞等）

**示例**：
```python
from grace_inference.utils import validate_structure

try:
    validate_structure(atoms)
    print("Structure is valid")
except ValueError as e:
    print(f"Invalid structure: {e}")
```

---

#### get_structure_info()

```python
get_structure_info(atoms: Atoms) -> dict
```

获取结构信息。

**参数**：
- `atoms` (Atoms): ASE Atoms 对象

**返回值**：dict，包含：
- `num_atoms` (int): 原子数
- `chemical_formula` (str): 化学式
- `volume` (float): 体积，单位 Å³
- `density` (float): 密度，单位 g/cm³
- `cell_parameters` (dict): 晶胞参数
- `pbc` (tuple): 周期性边界条件

**示例**：
```python
from grace_inference.utils import get_structure_info

info = get_structure_info(atoms)
print(f"Formula: {info['chemical_formula']}")
print(f"Atoms: {info['num_atoms']}")
print(f"Volume: {info['volume']:.2f} Å³")
print(f"Density: {info['density']:.3f} g/cm³")
```

---

## 4. CLI 命令行工具

GRACE Inference 提供完整的命令行界面，无需编写 Python 代码即可执行计算。

### 4.1 基本语法

```bash
grace-inference <command> [arguments] [options]
```

### 4.2 可用命令

#### single-point - 单点计算

```bash
grace-inference single-point <structure> [options]
```

**参数**：
- `<structure>`: 结构文件路径

**选项**：
- `--model <name>`: 模型名称，默认 `grace-2l`
- `--device <device>`: 计算设备 (`auto`, `cpu`, `cuda`)
- `--output <file>`: JSON 输出文件

**示例**：
```bash
# 基本用法
grace-inference single-point structure.cif

# 使用 GPU
grace-inference single-point structure.cif --model grace-3l --device cuda

# 保存结果
grace-inference single-point structure.cif --output result.json
```

---

#### optimize - 结构优化

```bash
grace-inference optimize <structure> [options]
```

**参数**：
- `<structure>`: 输入结构文件

**选项**：
- `--model <name>`: 模型名称
- `--device <device>`: 计算设备
- `--fmax <value>`: 力收敛标准，默认 0.05
- `--steps <n>`: 最大步数，默认 500
- `--optimizer <name>`: 优化算法 (`LBFGS`, `BFGS`, `FIRE`)
- `--cell`: 同时优化晶胞参数
- `--output <file>`: 输出结构文件
- `--trajectory <file>`: 轨迹文件

**示例**：
```bash
# 基本优化
grace-inference optimize input.cif --output optimized.cif

# 优化晶胞
grace-inference optimize input.cif --fmax 0.01 --cell --output optimized.cif

# 保存轨迹
grace-inference optimize input.cif --trajectory opt.traj --output optimized.cif
```

---

#### md - 分子动力学

```bash
grace-inference md <structure> [options]
```

**参数**：
- `<structure>`: 初始结构文件

**选项**：
- `--model <name>`: 模型名称
- `--device <device>`: 计算设备
- `--ensemble <type>`: MD 系综 (`nve`, `nvt`, `npt`)，默认 `nvt`
- `--temp <T>`: 温度（K），默认 300
- `--pressure <P>`: 压强（GPa），仅用于 NPT
- `--steps <n>`: MD 步数，默认 1000
- `--timestep <dt>`: 时间步长（fs），默认 1.0
- `--trajectory <file>`: 轨迹文件
- `--logfile <file>`: 日志文件

**示例**：
```bash
# NVT 动力学
grace-inference md init.cif --ensemble nvt --temp 300 --steps 10000 --trajectory md.traj

# NPT 动力学
grace-inference md init.cif --ensemble npt --temp 500 --pressure 1.0 --steps 20000

# 高温 NVE
grace-inference md init.cif --ensemble nve --temp 800 --steps 5000
```

---

#### phonon - 声子计算

```bash
grace-inference phonon <structure> [options]
```

**参数**：
- `<structure>`: 结构文件

**选项**：
- `--model <name>`: 模型名称
- `--device <device>`: 计算设备
- `--supercell <nx> <ny> <nz>`: 超胞尺寸，默认 `2 2 2`
- `--mesh <nx> <ny> <nz>`: k点网格，默认 `20 20 20`
- `--temp-range <Tmin> <Tmax> <Tstep>`: 温度范围（K）
- `--output-dir <dir>`: 输出目录

**示例**：
```bash
# 基本声子计算
grace-inference phonon structure.cif --supercell 3 3 3

# 计算热力学性质
grace-inference phonon structure.cif --supercell 3 3 3 --mesh 30 30 30 \
    --temp-range 0 1000 10 --output-dir phonon_results
```

---

#### bulk-modulus - 体积模量

```bash
grace-inference bulk-modulus <structure> [options]
```

**参数**：
- `<structure>`: 结构文件

**选项**：
- `--model <name>`: 模型名称
- `--device <device>`: 计算设备
- `--points <n>`: 体积采样点数，默认 11
- `--strain-range <value>`: 应变范围，默认 0.05
- `--eos <type>`: 状态方程 (`birchmurnaghan`, `vinet`, `murnaghan`)
- `--plot`: 生成 EOS 拟合图
- `--output-dir <dir>`: 输出目录

**示例**：
```bash
# 基本计算
grace-inference bulk-modulus structure.cif --plot

# 高精度计算
grace-inference bulk-modulus structure.cif --points 21 --strain-range 0.08 \
    --eos vinet --plot --output-dir bulk_results
```

---

#### adsorption - 吸附能计算

```bash
grace-inference adsorption <host> <adsorbate> <combined> [options]
```

**参数**：
- `<host>`: 主体结构文件
- `<adsorbate>`: 吸附质结构文件
- `<combined>`: 复合结构文件

**选项**：
- `--model <name>`: 模型名称
- `--device <device>`: 计算设备
- `--relax-all`: 优化所有结构
- `--no-relax-host`: 不优化主体
- `--no-relax-adsorbate`: 不优化吸附质
- `--no-relax-combined`: 不优化复合结构
- `--fmax <value>`: 力收敛标准，默认 0.05
- `--output <file>`: JSON 输出文件

**示例**：
```bash
# 基本吸附能计算
grace-inference adsorption MOF-5.cif CO2.xyz MOF-5_CO2.cif --relax-all

# 不优化主体（已优化）
grace-inference adsorption MOF-5_relaxed.cif H2.xyz MOF-5_H2.cif \
    --no-relax-host --fmax 0.01 --output adsorption_result.json
```

---

## 5. 数据结构

### 5.1 计算结果字典

#### 单点计算结果

```python
{
    'energy': float,              # 总能量 (eV)
    'energy_per_atom': float,     # 每原子能量 (eV/atom)
    'forces': np.ndarray,         # 原子受力 (N, 3) (eV/Å)
    'max_force': float,           # 最大力 (eV/Å)
    'rms_force': float,           # 均方根力 (eV/Å)
    'stress': np.ndarray,         # Voigt 应力 (6,) (eV/Å³)
    'pressure_GPa': float         # 压强 (GPa)
}
```

#### 声子计算结果

```python
{
    'phonon': Phonopy,                      # Phonopy 对象
    'supercell_matrix': list,               # 超胞矩阵
    'displacement': float,                  # 位移距离 (Å)
    'mesh': tuple,                          # k点网格
    'thermal_properties': {                 # 热力学性质（可选）
        'temperatures': np.ndarray,         # 温度 (K)
        'free_energy': np.ndarray,          # 自由能 (kJ/mol)
        'entropy': np.ndarray,              # 熵 (J/(K·mol))
        'heat_capacity': np.ndarray         # 热容 (J/(K·mol))
    }
}
```

#### 体积模量结果

```python
{
    'bulk_modulus_GPa': float,        # 体积模量 (GPa)
    'equilibrium_volume': float,      # 平衡体积 (Å³)
    'equilibrium_energy': float,      # 平衡能量 (eV)
    'eos_parameters': dict,           # EOS 参数
    'volumes': np.ndarray,            # 体积数组
    'energies': np.ndarray            # 能量数组
}
```

#### 吸附能结果

```python
{
    'adsorption_energy_eV': float,          # 吸附能 (eV)
    'adsorption_energy_kJ_mol': float,      # 吸附能 (kJ/mol)
    'host_energy': float,                   # 主体能量 (eV)
    'adsorbate_energy': float,              # 吸附质能量 (eV)
    'combined_energy': float,               # 复合结构能量 (eV)
    'host_optimized': Atoms,                # 优化的主体
    'adsorbate_optimized': Atoms,           # 优化的吸附质
    'combined_optimized': Atoms             # 优化的复合结构
}
```

---

## 6. 异常处理

### 6.1 常见异常

#### ImportError

```python
ImportError: GRACE not installed. Install with: pip install grace-calculator
```

**解决方案**：安装 GRACE 计算器包
```bash
pip install grace-calculator
```

---

#### ValueError (设备不可用)

```python
ValueError: CUDA requested but not available. Install GPU version of PyTorch or use device='cpu'
```

**解决方案**：
1. 安装 GPU 版本 PyTorch
2. 或使用 `device='cpu'` 或 `device='auto'`

---

#### FileNotFoundError

```python
FileNotFoundError: Structure file not found: structure.cif
```

**解决方案**：检查文件路径是否正确

---

#### ValueError (结构无效)

```python
ValueError: Atoms 0 and 1 are too close (0.234 Å). Possible overlapping atoms.
```

**解决方案**：检查并修正输入结构

---

### 6.2 错误处理示例

```python
from grace_inference import GRACEInference

try:
    calc = GRACEInference(device="cuda")
    result = calc.single_point("structure.cif")
except ImportError as e:
    print(f"Installation error: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
except FileNotFoundError as e:
    print(f"File error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## 附录 A: 单位转换

### A.1 能量单位

- 1 eV = 96.485 kJ/mol
- 1 eV = 23.061 kcal/mol
- 1 Ha = 27.211 eV

### A.2 力单位

- 1 eV/Å = 160.218 nN
- 1 eV/Å = 1.602 × 10⁻⁹ N

### A.3 压强单位

- 1 eV/Å³ = 160.218 GPa
- 1 GPa = 10 kbar
- 1 atm = 0.0001013 GPa

### A.4 温度单位

- T(K) = T(°C) + 273.15

---

## 附录 B: 推荐计算参数

### B.1 结构优化

| 应用场景 | fmax | optimizer | steps |
|---------|------|-----------|-------|
| 快速预优化 | 0.1 | FIRE | 200 |
| 标准优化 | 0.05 | LBFGS | 500 |
| 高精度优化 | 0.01 | LBFGS | 1000 |
| 晶胞优化 | 0.01 | LBFGS | 1000 |

### B.2 分子动力学

| 应用场景 | ensemble | 温度 (K) | 步长 (fs) | 总步数 |
|---------|----------|---------|----------|--------|
| 平衡化 | NVT | 300 | 1.0 | 5000 |
| 生产运行 | NVT | 300 | 1.0 | 50000+ |
| 高温退火 | NVT | 800 | 0.5 | 10000 |
| 等压模拟 | NPT | 300 | 1.0 | 50000+ |

### B.3 声子计算

| 体系规模 | supercell | mesh | 计算量 |
|---------|-----------|------|--------|
| 小 (<50 原子) | (3, 3, 3) | (30, 30, 30) | 高 |
| 中 (50-200 原子) | (2, 2, 2) | (20, 20, 20) | 中 |
| 大 (>200 原子) | (2, 2, 2) | (15, 15, 15) | 低 |

---

## 附录 C: 完整示例

### C.1 MOF 吸附能计算完整流程

```python
from grace_inference import GRACEInference

# 初始化计算器
calc = GRACEInference(model_name="grace-3l", device="cuda")

# 步骤 1: 优化 MOF 结构
print("Step 1: Optimizing MOF structure...")
mof_optimized = calc.optimize(
    "MOF-5.cif",
    fmax=0.01,
    optimize_cell=True,
    output="MOF-5_optimized.cif"
)

# 步骤 2: 计算吸附能
print("Step 2: Calculating adsorption energy...")
ads_result = calc.adsorption_energy(
    host_structure="MOF-5_optimized.cif",
    adsorbate_structure="CO2.xyz",
    combined_structure="MOF-5_CO2.cif",
    relax_host=False,  # 已优化
    fmax=0.01
)

print(f"Adsorption Energy: {ads_result['adsorption_energy_eV']:.4f} eV")
print(f"Adsorption Energy: {ads_result['adsorption_energy_kJ_mol']:.2f} kJ/mol")

# 步骤 3: 分子动力学验证
print("Step 3: Running MD simulation...")
final = calc.molecular_dynamics(
    ads_result['combined_optimized'],
    ensemble="nvt",
    temperature_K=300,
    steps=10000,
    trajectory="MOF-5_CO2_md.traj"
)
```

---

**文档版本**: 1.0.0  
**最后更新**: 2026年1月7日  
**许可证**: MIT License
