# MACE 推理任务详细接口文档

本文档详细说明每个推理任务的输入输出、调用接口和物理意义。所有接口已验证在以下版本中可用：
- ASE >= 3.22.0 (建议 >= 3.23.0)
- Phonopy >= 2.20.0
- mace-torch >= 0.3.10

---

## 1. 静态建模任务

### 1.1 单点能量计算

**物理意义**：
计算给定几何构型的总能量，这是所有量子化学和分子模拟的基础。单点能量可用于：
- 评估结构稳定性（能量越低越稳定）
- 比较不同构型的相对稳定性
- 计算反应能、吸附能等能量差
- 作为后续优化、动力学的起点

**输入**：
- `atoms`: `ase.Atoms` 对象，包含原子坐标、元素类型、晶胞参数
- `calculator`: MACE 计算器对象，已加载模型

**输出**：
- `energy`: `float`，体系总能量，单位：eV

**调用接口**：
```python
# ASE >= 3.22.0
energy = atoms.get_potential_energy()  # 返回 float，单位 eV
```

**示例代码**：
```python
from ase.io import read
from mace.calculators import mace_mp

# 输入：CIF 文件
atoms = read('MOF.cif')

# 设置计算器
calculator = mace_mp(model="medium", device='cuda')
atoms.calc = calculator

# 输出：总能量
energy = atoms.get_potential_energy()
print(f"Total Energy: {energy:.6f} eV")
```

---

### 1.2 原子受力计算

**物理意义**：
计算每个原子受到的力（能量对坐标的负梯度），用于：
- 判断结构是否处于平衡态（力接近零表示平衡）
- 结构优化（沿负力方向移动原子）
- 分子动力学（根据力计算加速度）

**输入**：
- `atoms`: `ase.Atoms` 对象

**输出**：
- `forces`: `numpy.ndarray`，形状 `(N, 3)`，N 为原子数，单位：eV/Å

**调用接口**：
```python
# ASE >= 3.22.0
forces = atoms.get_forces()  # 返回 np.ndarray，形状 (N_atoms, 3)
```

---

### 1.3 应力张量计算

**物理意义**：
计算晶胞受到的应力（能量对应变的负导数），用于：
- 判断晶胞是否处于力学平衡（应力接近零）
- 晶胞优化（调整晶胞参数）
- 计算弹性常数、体积模量

**输入**：
- `atoms`: `ase.Atoms` 对象

**输出**：
- `stress`: `numpy.ndarray`
  - Voigt 形式（默认）：形状 `(6,)`，顺序 `[σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]`
  - 3×3 矩阵形式：形状 `(3, 3)`
  - 单位：eV/Å³

**调用接口**：
```python
# ASE >= 3.22.0
stress_voigt = atoms.get_stress(voigt=True)   # 返回 (6,) 数组
stress_matrix = atoms.get_stress(voigt=False) # 返回 (3, 3) 数组
```

---

### 1.4 结构优化

**物理意义**：
调整原子坐标和/或晶胞参数，使能量最小化、力和应力趋于零，得到稳定平衡构型。

**输入**：
- `atoms`: `ase.Atoms` 对象
- `optimizer`: 优化器类，如 `LBFGS` 或 `BFGS`
- `filter`: 晶胞过滤器（如需优化晶胞），ASE >= 3.23.0 推荐使用 `FrechetCellFilter`
- `fmax`: `float`，收敛标准（原子受力最大值），默认 0.05 eV/Å
- `steps`: `int`，最大优化步数

**输出**：
- 优化后的 `atoms` 对象（原地修改）
- `converged`: `bool`，优化器的 `run()` 返回是否收敛

**调用接口**：
```python
# ASE >= 3.23.0 推荐使用 FrechetCellFilter
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

# 仅优化原子坐标
optimizer = LBFGS(atoms, trajectory='opt.traj')
optimizer.run(fmax=0.05)

# 同时优化原子坐标和晶胞
filtered_atoms = FrechetCellFilter(atoms)
optimizer = LBFGS(filtered_atoms, trajectory='opt.traj')
optimizer.run(fmax=0.05)
```

**注意**：
- ASE < 3.23.0 使用 `ExpCellFilter`，但该类已废弃
- `FrechetCellFilter` 位于 `ase.filters` 或 `ase.constraints`

---

## 2. 动力学模拟任务

### 2.1 正则系综 (NVT) 分子动力学

**物理意义**：
在恒定原子数(N)、体积(V)、温度(T)下进行分子动力学模拟，使用 Langevin 恒温器控温。用于：
- 研究材料在特定温度下的热运动
- 计算热力学性质（扩散系数、径向分布函数等）
- 采样构型空间

**输入**：
- `atoms`: `ase.Atoms` 对象
- `timestep`: `float`，时间步长，单位：fs，典型值 0.5-2 fs
- `temperature_K`: `float`，目标温度，单位：K
- `friction`: `float`，摩擦系数，单位：1/fs，典型值 0.002-0.01

**输出**：
- 轨迹文件（通过 `MDLogger` 或 `Trajectory` 保存）
- 每一步的能量、温度、坐标等

**调用接口**：
```python
# ASE >= 3.22.0
from ase.md.langevin import Langevin
from ase import units

dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,      # 时间步长，转换为内部单位
    temperature_K=300,             # 目标温度，单位 K
    friction=0.002 / units.fs      # 摩擦系数，单位 1/fs
)
dyn.run(1000)  # 运行 1000 步
```

**参数说明**：
- `temperature_K`: 直接指定开尔文温度，无需单位转换
- `units.fs`: ASE 的时间单位转换常数（1 fs = 1 × units.fs）

---

### 2.2 等温等压系综 (NPT) 分子动力学

**物理意义**：
在恒定原子数(N)、压强(P)、温度(T)下进行分子动力学模拟，晶胞体积可变。用于：
- 研究材料在特定温度和压强下的平衡结构
- 计算热膨胀系数
- 模拟实际实验条件（常压或高压）

**输入**：
- `atoms`: `ase.Atoms` 对象
- `timestep`: `float`，时间步长，单位：fs
- `temperature_K`: `float`，目标温度，单位：K
- `pressure_au`: `float`，目标压强，原子单位（需转换）
- `taut`: `float`，温度弛豫时间，单位：fs
- `taup`: `float`，压强弛豫时间，单位：fs
- `compressibility_au`: `float`，等温压缩系数，原子单位

**输出**：
- 优化后的晶胞参数和原子坐标
- 轨迹文件

**调用接口**：
```python
# ASE >= 3.22.0
from ase.md.nptberendsen import NPTBerendsen
from ase import units

# 压强转换：1 GPa = 1 GPa * units.GPa （转为原子单位）
pressure_au = 1.0 * units.GPa

dyn = NPTBerendsen(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    pressure_au=pressure_au,        # 使用 pressure_au，不是 pressure
    taut=100 * units.fs,
    taup=1000 * units.fs,
    compressibility_au=4.57e-5 / units.GPa
)
dyn.run(5000)
```

**注意**：
- 参数名为 `pressure_au`，不是 `pressure`（后者已废弃）
- 压缩系数典型值：4.57e-5 / GPa（需转换为原子单位）

---

## 3. 声子与热力学性质

### 3.1 声子色散计算

**物理意义**：
计算晶格振动模式（声子）及其频率，用于：
- 判断结构动力学稳定性（负频率表示不稳定）
- 理解材料的振动特性
- 计算热力学性质（自由能、熵、热容）

**输入**：
- `atoms`: `ase.Atoms` 对象（原胞或超胞）
- `supercell_matrix`: `list` 或 `numpy.ndarray`，超胞尺寸，如 `[2, 2, 2]`
- `distance`: `float`，原子位移距离，单位：Å，典型值 0.01-0.03 Å

**输出**：
- `phonon`: `phonopy.Phonopy` 对象，包含力常数矩阵
- 声子频率和本征矢（通过 `phonon.get_band_structure()` 等获取）

**调用接口**：
```python
# Phonopy >= 2.20.0
from phonopy import Phonopy

# 创建 Phonopy 对象
phonon = Phonopy(atoms, supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])

# 生成位移
phonon.generate_displacements(distance=0.01)
disp_supercells = phonon.supercells_with_displacements

# 计算每个位移超胞的力
for scell in disp_supercells:
    scell.calc = calculator
    forces = scell.get_forces()
    phonon.forces.append(forces)  # 手动添加

# 或使用 set_forces()
phonon.set_forces(forces_list)  # forces_list 为所有位移的力列表

# 计算力常数
phonon.produce_force_constants()
```

---

### 3.2 热力学性质计算

**物理意义**：
基于声子态密度，计算有限温度下的热力学性质：
- **自由能 (F)**：衡量体系稳定性
- **熵 (S)**：体系的无序度
- **热容 (C_v)**：温度变化时吸收的热量

这些性质对于理解材料的相变、热稳定性至关重要。

**输入**：
- `phonon`: `phonopy.Phonopy` 对象（已计算力常数）
- `mesh`: `list`，声子态密度的 k 点网格，如 `[20, 20, 20]`
- `t_min`: `float`，最低温度，单位：K
- `t_max`: `float`，最高温度，单位：K
- `t_step`: `float`，温度步长，单位：K

**输出**：
- `thermal_properties`: `dict`，包含以下键值：
  - `'temperatures'`: `numpy.ndarray`，温度数组，单位：K
  - `'free_energy'`: `numpy.ndarray`，自由能，单位：kJ/mol
  - `'entropy'`: `numpy.ndarray`，熵，单位：J/(mol·K)
  - `'heat_capacity'`: `numpy.ndarray`，热容 C_v，单位：J/(mol·K)

**调用接口**：
```python
# Phonopy >= 2.20.0
# 计算声子态密度
phonon.run_mesh([20, 20, 20])

# 计算热力学性质
phonon.run_thermal_properties(t_step=10, t_max=1000, t_min=0)

# 获取结果字典
tp_dict = phonon.get_thermal_properties_dict()
temperatures = tp_dict['temperatures']      # 单位：K
free_energy = tp_dict['free_energy']        # 单位：kJ/mol
entropy = tp_dict['entropy']                # 单位：J/(mol·K)
heat_capacity = tp_dict['heat_capacity']    # 单位：J/(mol·K)
```

---

## 4. 力学性质

### 4.1 体积模量计算

**物理意义**：
体积模量 (Bulk Modulus, B₀) 衡量材料抵抗均匀压缩的能力：
- B₀ 越大，材料越难被压缩（越"硬"）
- 用于评估材料的力学强度、抗压性能
- 是弹性常数张量的重要组成部分
- 对于 MOF 材料，体积模量反映骨架的刚性

**输入**：
- `atoms`: `ase.Atoms` 对象
- `volumes`: `list` 或 `numpy.ndarray`，不同体积下的晶胞体积，单位：Å³
- `energies`: `list` 或 `numpy.ndarray`，对应的能量，单位：eV
- `eos`: `str`，状态方程类型，如 `'birchmurnaghan'`（Birch-Murnaghan 方程）

**输出**：
- `v0`: `float`，平衡体积，单位：Å³
- `e0`: `float`，平衡能量，单位：eV
- `B`: `float`，体积模量，单位：eV/Å³（需转换为 GPa）

**调用接口**：
```python
# ASE >= 3.22.0
from ase.eos import EquationOfState
import numpy as np

# 准备体积-能量数据（通过缩放晶胞得到）
volumes = []
energies = []
for scale in np.linspace(0.95, 1.05, 11):
    cell_scaled = atoms.get_cell() * scale
    atoms_scaled = atoms.copy()
    atoms_scaled.set_cell(cell_scaled, scale_atoms=True)
    atoms_scaled.calc = calculator
    volumes.append(atoms_scaled.get_volume())
    energies.append(atoms_scaled.get_potential_energy())

# 拟合状态方程
eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
v0, e0, B = eos.fit()  # B 单位：eV/Å³

# 转换为 GPa
B_GPa = B * 160.21766208  # 转换因子
print(f"Bulk Modulus: {B_GPa:.2f} GPa")
```

**单位转换**：
- 1 eV/Å³ = 160.21766208 GPa

---

## 5. 配位环境稳定性

### 5.1 金属-配体配位键长分析

**物理意义**：
分析金属中心与配体原子之间的键长，用于：
- 评估配位环境的稳定性（键长适中表示稳定）
- 比较不同金属的配位特性
- 研究吸附、催化反应中的键合变化

**输入**：
- `atoms`: `ase.Atoms` 对象
- `metal_indices`: `list`，金属原子索引
- `cutoff`: `float`，搜索配位原子的截断半径，单位：Å

**输出**：
- `coordination_analysis`: `dict`，包含每个金属的配位数、配位原子索引、键长列表

**调用接口**：
```python
# ASE >= 3.22.0
from ase.neighborlist import NeighborList, natural_cutoffs

# 方法 1：使用 natural_cutoffs 自动生成截断半径
cutoffs = natural_cutoffs(atoms, mult=1.0)  # mult 可调整
nl = NeighborList(cutoffs, skin=0.3, self_interaction=False, bothways=False)
nl.update(atoms)

# 方法 2：手动指定截断半径
cutoffs = [3.0] * len(atoms)  # 所有原子 3.0 Å
nl = NeighborList(cutoffs, skin=0.3, self_interaction=False, bothways=False)
nl.update(atoms)

# 获取邻居
for metal_idx in metal_indices:
    indices, offsets = nl.get_neighbors(metal_idx)
    # indices: 邻居原子索引数组
    # offsets: 晶胞偏移数组（用于周期性边界）
    
    for neighbor_idx, offset in zip(indices, offsets):
        # 计算键长（考虑周期性边界）
        pos_metal = atoms.positions[metal_idx]
        pos_neighbor = atoms.positions[neighbor_idx] + offset @ atoms.cell
        distance = np.linalg.norm(pos_neighbor - pos_metal)
```

**参数说明**：
- `natural_cutoffs(atoms, mult=1.0)`: 基于共价半径自动生成截断半径
- `mult`: 乘数因子，增大可搜索更远的邻居
- `skin`: 额外的安全距离，用于优化更新频率
- `self_interaction`: 是否包含自身
- `bothways`: 是否双向记录（A-B 和 B-A）

---

## 6. 主客体相互作用

### 6.1 气体分子吸附能计算

**物理意义**：
计算气体分子在 MOF 孔道内的吸附能（结合能），用于：
- 评估 MOF 对特定气体的吸附能力
- 筛选高性能气体存储/分离材料
- 理解吸附位点和相互作用机制

**吸附能定义**：
```
E_ads = E(MOF+gas) - E(MOF) - E(gas)
```
- E_ads < 0 表示吸附放热（稳定吸附）
- |E_ads| 越大，吸附越强

**输入**：
- `MOF_atoms`: `ase.Atoms` 对象，MOF 结构
- `gas_molecule`: `str` 或 `ase.Atoms`，气体分子名称（如 `'H2O'`, `'CO2'`）或已构建的分子对象
- `adsorption_site`: 吸附位点坐标（可通过几何分析或优化确定）

**输出**：
- `E_ads`: `float`，吸附能，单位：eV

**调用接口**：
```python
# ASE >= 3.22.0
from ase.build import molecule

# 1. 计算纯 MOF 能量
MOF_atoms.calc = calculator
E_MOF = MOF_atoms.get_potential_energy()

# 2. 构建气体分子（使用 ASE 内置分子库）
gas = molecule('H2O')  # 支持 'H2O', 'CO2', 'CH4', 'N2', 'O2' 等
gas.calc = calculator
E_gas = gas.get_potential_energy()

# 3. 构建吸附复合物
complex_atoms = MOF_atoms.copy()
# 将气体分子平移到吸附位点
gas.translate(adsorption_site - gas.get_center_of_mass())
complex_atoms += gas  # 合并结构

# 优化吸附构型
complex_atoms.calc = calculator
optimizer = LBFGS(complex_atoms)
optimizer.run(fmax=0.05)
E_complex = complex_atoms.get_potential_energy()

# 4. 计算吸附能
E_ads = E_complex - E_MOF - E_gas
print(f"Adsorption Energy: {E_ads:.3f} eV")
```

**ASE 内置分子库**（部分）：
- 小分子：`'H2'`, `'N2'`, `'O2'`, `'CO'`, `'CO2'`, `'NO'`, `'NO2'`
- 水：`'H2O'`
- 烃类：`'CH4'`, `'C2H6'`, `'C2H4'`, `'C2H2'`
- 完整列表可通过 `ase.build.molecule.names` 查看

---

## 7. 版本兼容性总结

| 接口 | 所属库 | 最低版本 | 推荐版本 | 备注 |
|-----|--------|---------|---------|------|
| `Atoms.get_potential_energy()` | ASE | 3.22.0 | 3.23.0+ | 基础接口 |
| `Atoms.get_forces()` | ASE | 3.22.0 | 3.23.0+ | 返回 (N, 3) 数组 |
| `Atoms.get_stress(voigt=True)` | ASE | 3.22.0 | 3.23.0+ | Voigt 或 3×3 矩阵 |
| `FrechetCellFilter` | ASE | 3.23.0 | 3.23.0+ | **推荐**，替代 ExpCellFilter |
| `ExpCellFilter` | ASE | 3.22.0 | - | **已废弃**（3.23.0+） |
| `LBFGS`, `BFGS` | ASE | 3.22.0 | 3.23.0+ | 优化器 |
| `Langevin(temperature_K=...)` | ASE | 3.22.0 | 3.23.0+ | NVT 分子动力学 |
| `NPTBerendsen(pressure_au=...)` | ASE | 3.22.0 | 3.23.0+ | 使用 `pressure_au`，不是 `pressure` |
| `EquationOfState` | ASE | 3.22.0 | 3.23.0+ | B 单位 eV/Å³ |
| `NeighborList` | ASE | 3.22.0 | 3.23.0+ | 配位分析 |
| `natural_cutoffs(atoms)` | ASE | 3.22.0 | 3.23.0+ | 自动生成截断半径 |
| `ase.build.molecule('H2O')` | ASE | 3.22.0 | 3.23.0+ | 内置分子库 |
| `Phonopy` | Phonopy | 2.20.0 | 2.20.0+ | 声子计算 |
| `phonon.generate_displacements()` | Phonopy | 2.20.0 | 2.20.0+ | 生成位移 |
| `phonon.produce_force_constants()` | Phonopy | 2.20.0 | 2.20.0+ | 计算力常数 |
| `phonon.run_mesh()` | Phonopy | 2.20.0 | 2.20.0+ | 声子态密度 |
| `phonon.run_thermal_properties()` | Phonopy | 2.20.0 | 2.20.0+ | 热力学性质 |
| `phonon.get_thermal_properties_dict()` | Phonopy | 2.20.0 | 2.20.0+ | 返回字典格式结果 |

**关键废弃提示**：
- ✅ 使用 `FrechetCellFilter`（ASE >= 3.23.0）
- ❌ 避免 `ExpCellFilter`（已废弃）
- ✅ 使用 `pressure_au` 参数
- ❌ 避免 `pressure` 参数（已废弃）

---

## 8. 常见单位转换

| 物理量 | ASE 内部单位 | 常用单位 | 转换因子（ASE → 常用） |
|--------|------------|---------|---------------------|
| 能量 | eV | eV | 1.0 |
| 力 | eV/Å | eV/Å | 1.0 |
| 应力 | eV/Å³ | GPa | 160.21766208 |
| 体积模量 | eV/Å³ | GPa | 160.21766208 |
| 时间 | 内部时间单位 | fs | `1.0 * units.fs` |
| 温度 | K | K | 1.0（直接指定） |
| 压强 | eV/Å³ | GPa | `1.0 * units.GPa` |

**示例**：
```python
from ase import units

# 时间：1 fs
timestep = 1.0 * units.fs

# 压强：1 GPa
pressure = 1.0 * units.GPa

# 应力：eV/Å³ → GPa
stress_GPa = stress_eV_A3 * 160.21766208

# 体积模量：eV/Å³ → GPa
B_GPa = B_eV_A3 * 160.21766208
```
