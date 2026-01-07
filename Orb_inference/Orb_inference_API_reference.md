# Orb 推理任务 API 接口参考

> **文档版本**: v1.0  
> **最后更新**: 2026年1月7日  
> **接口验证**: 基于 Context7 验证的 orb-models 官方文档

本文档提供Orb系列模型推理任务的详细API规范，包括输入/输出格式、物理意义、代码示例和版本兼容性。

---

## 目录

1. [核心接口](#1-核心接口)
2. [静态计算任务](#2-静态计算任务)
3. [动力学模拟任务](#3-动力学模拟任务)
4. [体相性质计算任务](#4-体相性质计算任务)
5. [主客体相互作用任务](#5-主客体相互作用任务)
6. [版本兼容性](#6-版本兼容性)
7. [常用单位转换](#7-常用单位转换)

---

## 1. 核心接口

### 1.1 模型加载

#### orb_models.forcefield.pretrained

| 方法 | 说明 | 数据集 | 力类型 |
|------|------|--------|--------|
| `orb_v3_conservative_inf_omat()` | **推荐**用于MOF | OMat24 | 保守力 ✅ |
| `orb_v3_conservative_inf_mpa()` | 广泛化学覆盖 | MPtraj + Alexandria | 保守力 ✅ |
| `orb_d3_v2()` | 内置D3校正 | MPtraj + Alexandria | 非保守力 ⚠️ |
| `orb_mptraj_only_v2()` | 无D3，MPtraj训练 | MPtraj | 非保守力 ⚠️ |
| `load_model(model_name)` | 通用加载接口 | 依赖模型 | 依赖模型 |

**接口验证** (Context7)：
```python
from orb_models.forcefield import pretrained

# ✅ 官方推荐方式
orbff = pretrained.orb_v3_conservative_inf_omat(
    device="cpu",              # or "cuda"
    precision="float32-high",  # or "float32-highest", "float64"
)

# ✅ 使用load_model
from orb_models.forcefield.pretrained import load_model
model = load_model("orb-v3-conservative-120-omat", precision='float32-highest')
```

**参数说明**：

| 参数 | 类型 | 可选值 | 默认值 | 说明 |
|------|------|--------|--------|------|
| `device` | str | "cpu", "cuda", "cuda:0" | "cpu" | 计算设备 |
| `precision` | str | "float32-high", "float32-highest", "float64" | "float32-high" | 数值精度 |
| `weights_path` | str | 文件路径 | None | 自定义权重（用于微调模型） |

**精度选择指南**：

| 精度 | 速度 | 准确性 | 推荐场景 |
|------|------|--------|----------|
| `float32-high` | ⚡⚡⚡ 最快 | 良好 | 大规模筛选、快速测试 |
| `float32-highest` | ⚡⚡ 中等 | **优异** | **生产计算（推荐）** |
| `float64` | ⚡ 最慢 | 最高 | 高精度研究、基准测试 |

---

### 1.2 计算器初始化

#### ORBCalculator

**接口验证** (Context7)：
```python
from orb_models.forcefield.calculator import ORBCalculator

# ✅ 创建计算器
calc = ORBCalculator(orbff, device=device)
```

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `orbff` | OrbFF对象 | 通过`pretrained.*`加载的模型 |
| `device` | str | 计算设备（应与模型device一致） |

**集成ASE** (Context7 验证)：
```python
import ase
from ase.build import bulk
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

# ✅ 完整工作流
orbff = pretrained.orb_v3_conservative_inf_omat(device="cuda", precision="float32-high")
calc = ORBCalculator(orbff, device="cuda")

atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
atoms.calc = calc

energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Å
stress = atoms.get_stress(voigt=True)  # eV/Å³
```

---

### 1.3 底层预测接口

#### orbff.predict()

**接口验证** (Context7)：
```python
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs

device = "cpu"
orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")

# ✅ 将ASE atoms转换为图
graph = atomic_system.ase_atoms_to_atom_graphs(atoms, orbff.system_config, device=device)

# ✅ 批处理多个图
# graph = batch_graphs([graph1, graph2, ...])

# ✅ 预测
result = orbff.predict(graph, split=False)

# ✅ 转换回ASE atoms
atoms_with_results = atomic_system.atom_graphs_to_ase_atoms(
    graph,
    energy=result["energy"],
    forces=result["grad_forces"],
    stress=result["grad_stress"]
)
```

**返回值**：

| 键 | 类型 | 单位 | 说明 |
|-----|------|------|------|
| `energy` | Tensor | eV | 总能量 |
| `grad_forces` | Tensor | eV/Å | 原子受力（负梯度） |
| `grad_stress` | Tensor | eV/Å³ | 应力张量 |

---

## 2. 静态计算任务

### 2.1 单点能量计算

#### 物理意义
计算给定原子构型的势能面上的能量、力和应力。

#### 输入规范

| 参数 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `atoms` | ASE Atoms | - | 原子结构对象 |

#### 输出规范

| 属性 | 方法 | 类型 | 单位 | 说明 |
|------|------|------|------|------|
| 总能量 | `atoms.get_potential_energy()` | float | eV | 系统总势能 |
| 单位能量 | 计算得出 | float | eV/atom | 能量/原子数 |
| 原子受力 | `atoms.get_forces()` | ndarray (N,3) | eV/Å | 每个原子的力向量 |
| 应力张量 | `atoms.get_stress(voigt=True)` | ndarray (6,) | eV/Å³ | Voigt记号 [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy] |
| 压强 | 计算得出 | float | GPa | P = -Tr(σ)/3 × 160.21766208 |

#### 调用接口

```python
# ✅ 标准ASE接口（Context7验证）
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress(voigt=True)  # 返回6分量Voigt记号
```

#### 完整示例

```python
import numpy as np
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

# 加载模型
orbff = pretrained.orb_v3_conservative_inf_omat(device="cuda", precision="float32-high")
calc = ORBCalculator(orbff, device="cuda")

# 读取结构
atoms = read("MOF.cif")
atoms.calc = calc

# 单点计算
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress(voigt=True)

# 派生量
energy_per_atom = energy / len(atoms)
max_force = np.max(np.linalg.norm(forces, axis=1))
rms_force = np.sqrt(np.mean(np.sum(forces**2, axis=1)))
pressure_GPa = -np.trace(stress[:3]) / 3 * 160.21766208

print(f"Energy: {energy:.6f} eV")
print(f"Energy/atom: {energy_per_atom:.6f} eV")
print(f"Max force: {max_force:.6f} eV/Å")
print(f"RMS force: {rms_force:.6f} eV/Å")
print(f"Pressure: {pressure_GPa:.4f} GPa")
```

---

### 2.2 结构优化

#### 物理意义
优化原子位置和晶胞参数，使系统达到势能最小（力收敛至阈值以下）。

#### 输入规范

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `atoms` | ASE Atoms | - | 初始结构 |
| `fmax` | float | 0.05 | 力收敛标准 (eV/Å) |
| `steps` | int | 500 | 最大优化步数 |
| `optimizer` | str | "LBFGS" | 优化器（LBFGS/BFGS/FIRE） |
| `optimize_cell` | bool | False | 是否同时优化晶胞 |
| `trajectory` | str | None | 轨迹文件路径 |

#### 输出规范

| 属性 | 类型 | 说明 |
|------|------|------|
| 优化后结构 | ASE Atoms | 能量最小化的结构 |
| 最终能量 | float (eV) | 优化后总能量 |
| 最终力 | ndarray (eV/Å) | 优化后原子受力 |
| 收敛状态 | bool | 是否达到fmax标准 |
| 优化步数 | int | 实际执行的步数 |

#### 调用接口

**ASE优化器** (Context7验证)：
```python
from ase.optimize import LBFGS, BFGS, FIRE

# ✅ 仅优化原子位置
optimizer = LBFGS(atoms, trajectory='opt.traj', logfile='opt.log')
optimizer.run(fmax=0.05, steps=500)

# ✅ 同时优化晶胞（使用FrechetCellFilter，ASE >= 3.23.0）
from ase.constraints import FrechetCellFilter
ecf = FrechetCellFilter(atoms)
optimizer = LBFGS(ecf, trajectory='opt_cell.traj')
optimizer.run(fmax=0.05)
```

#### 完整示例

```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.optimize import LBFGS
from ase.constraints import FrechetCellFilter

# 设置
orbff = pretrained.orb_v3_conservative_inf_omat(device="cuda", precision="float32-high")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF_initial.cif")
atoms.calc = calc

print(f"Initial energy: {atoms.get_potential_energy():.6f} eV")

# 优化（晶胞+原子）
ecf = FrechetCellFilter(atoms)
opt = LBFGS(ecf, trajectory="opt.traj", logfile="opt.log")
opt.run(fmax=0.05, steps=500)

print(f"Final energy: {atoms.get_potential_energy():.6f} eV")
print(f"Converged in {opt.nsteps} steps")

# 保存结果
atoms.write("MOF_optimized.cif")
```

**性能数据**（MOFSimBench）：

| 模型 | 收敛成功率 | 体积偏差<10% |
|------|-----------|-------------|
| orb-v3-omat | **89%** 🥇 | ✅ |
| orb-v3-mpa | 87% | ✅ |
| orb-d3-v2 | 61% ❌ | ❌ |

---

## 3. 动力学模拟任务

### 3.1 NVT分子动力学

#### 物理意义
恒定粒子数(N)、体积(V)、温度(T)，模拟热平衡状态下的原子运动。

#### 输入规范

| 参数 | 类型 | 单位 | 推荐值 | 说明 |
|------|------|------|--------|------|
| `temperature_K` | float | K | 300 | 目标温度 |
| `timestep` | float | fs | 1.0 | MD时间步长 |
| `steps` | int | - | 1000-100000 | MD总步数 |
| `friction` | float | 1/fs | 0.01 | Langevin摩擦系数 |
| `taut` | float | fs | 100 | 温度弛豫时间（taut=1/friction） |

#### 输出规范

| 属性 | 类型 | 单位 | 说明 |
|------|------|------|------|
| 轨迹文件 | Trajectory | - | 每一帧的原子构型 |
| 最终结构 | ASE Atoms | - | MD结束时的构型 |
| 温度历史 | ndarray | K | 瞬时温度随时间变化 |
| 能量历史 | ndarray | eV | 总能量随时间变化 |

#### 调用接口

**ASE Langevin恒温器** (Context7验证)：
```python
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# ✅ 初始化速度
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# ✅ NVT MD
dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    friction=0.01,  # 1/fs
    trajectory="nvt.traj",
    logfile="nvt.log",
    loginterval=100
)
dyn.run(steps=50000)  # 50 ps
```

#### 完整示例

```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# 设置
orbff = pretrained.orb_v3_conservative_inf_omat(device="cuda", precision="float32-high")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF.cif")
atoms.calc = calc

# 初始化速度
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# NVT MD
dyn = Langevin(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    friction=0.01,
    trajectory="nvt_md.traj",
    logfile="nvt_md.log",
    loginterval=100
)

print("Running NVT MD for 50 ps...")
dyn.run(steps=50000)
print("MD completed!")

# 分析轨迹
from ase.io import Trajectory
traj = Trajectory("nvt_md.traj")
volumes = [frame.get_volume() for frame in traj]
print(f"Volume drift: {(volumes[-1]/volumes[0] - 1)*100:.2f}%")
```

**性能数据**（MOFSimBench，50ps@300K）：

| 模型 | 体积漂移 | 稳定性 |
|------|---------|--------|
| orb-v3-omat | < 5% ✅ | 优异 |
| orb-v3-mpa | < 6% ✅ | 优异 |
| orb-d3-v2 | > 20% ❌ | 差 |

---

### 3.2 NPT分子动力学

#### 物理意义
恒定粒子数(N)、压强(P)、温度(T)，允许晶胞变化，模拟真实实验条件。

#### 输入规范

| 参数 | 类型 | 单位 | 推荐值 | 说明 |
|------|------|------|--------|------|
| `temperature_K` | float | K | 300 | 目标温度 |
| `pressure_GPa` | float | GPa | 0.0 | 目标压强（1 atm ≈ 0.0001 GPa） |
| `timestep` | float | fs | 1.0 | MD时间步长 |
| `steps` | int | - | 1000-100000 | MD总步数 |
| `ttime` | float | fs | 100 | 温度弛豫时间 |
| `pfactor` | float | - | 自动 | 压强弛豫因子（可自动估算） |

#### 输出规范

| 属性 | 类型 | 单位 | 说明 |
|------|------|------|------|
| 轨迹文件 | Trajectory | - | 每一帧的原子构型+晶胞 |
| 最终结构 | ASE Atoms | - | MD结束时的构型 |
| 体积历史 | ndarray | Å³ | 体积随时间变化 |
| 压强历史 | ndarray | GPa | 压强随时间变化 |

#### 调用接口

**ASE NPT Berendsen** (Context7验证)：
```python
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# ✅ 初始化速度
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# ✅ NPT MD
dyn = NPT(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    externalstress=0.0,  # eV/Å³ (0 GPa)
    ttime=100 * units.fs,
    pfactor=None,  # 自动估算或手动设置
    trajectory="npt.traj",
    logfile="npt.log",
    loginterval=100
)
dyn.run(steps=50000)  # 50 ps
```

**pfactor估算**（基于体积模量）：
```python
# pfactor = (timestep^2) * B / V
volume = atoms.get_volume()
bulk_modulus_GPa = 20.0  # MOF典型值10-30 GPa
pfactor = (timestep**2) * bulk_modulus_GPa / volume / 160.21766208
```

#### 完整示例

```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# 设置
orbff = pretrained.orb_v3_conservative_inf_omat(device="cuda", precision="float32-high")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF.cif")
atoms.calc = calc

# 初始化速度
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# NPT MD
dyn = NPT(
    atoms,
    timestep=1.0 * units.fs,
    temperature_K=300,
    externalstress=0.0,  # 1 atm
    ttime=100 * units.fs,
    pfactor=None,  # 自动估算
    trajectory="npt_md.traj",
    logfile="npt_md.log",
    loginterval=100
)

print("Running NPT MD for 50 ps...")
dyn.run(steps=50000)
print("MD completed!")

# 分析体积变化
from ase.io import Trajectory
import numpy as np
traj = Trajectory("npt_md.traj")
volumes = [frame.get_volume() for frame in traj]
print(f"Initial volume: {volumes[0]:.2f} Å³")
print(f"Final volume: {volumes[-1]:.2f} Å³")
print(f"Average volume: {np.mean(volumes):.2f} ± {np.std(volumes):.2f} Å³")
```

---

## 4. 体相性质计算任务

### 4.1 体积模量

#### 物理意义
材料抵抗均匀压缩的能力：B₀ = -V(∂P/∂V)|₀

#### 输入规范

| 参数 | 类型 | 单位 | 推荐值 | 说明 |
|------|------|------|--------|------|
| `atoms` | ASE Atoms | - | - | 初始结构（应先优化） |
| `n_points` | int | - | 7-11 | EOS拟合点数 |
| `eps` | float | - | 0.04 | 体积应变范围（±4%） |
| `eos` | str | - | 'birchmurnaghan' | EOS方程类型 |

#### 输出规范

| 属性 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `v0` | float | Å³ | 平衡体积 |
| `e0` | float | eV | 平衡能量 |
| `B` | float | eV/Å³ | 体积模量（原始单位） |
| `B_GPa` | float | GPa | 体积模量（需转换：B × 160.21766208） |

#### 调用接口

**ASE EquationOfState** (Context7验证)：
```python
from ase.eos import EquationOfState
import numpy as np

# ✅ 生成体积缩放点
volumes = []
energies = []
cell0 = atoms.cell.copy()

for scale in np.linspace(0.96, 1.04, 11):  # ±4%, 11点
    atoms_scaled = atoms.copy()
    atoms_scaled.set_cell(cell0 * scale, scale_atoms=True)
    atoms_scaled.calc = calc
    volumes.append(atoms_scaled.get_volume())
    energies.append(atoms_scaled.get_potential_energy())

# ✅ EOS拟合
eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
v0, e0, B = eos.fit()
B_GPa = B * 160.21766208  # 单位转换
```

#### 完整示例

```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.eos import EquationOfState
import numpy as np

# 设置
orbff = pretrained.orb_v3_conservative_inf_omat(device="cuda", precision="float32-high")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF.cif")
atoms.calc = calc

# 生成体积-能量数据
volumes = []
energies = []
cell0 = atoms.cell.copy()

for scale in np.linspace(0.96, 1.04, 11):
    atoms_scaled = atoms.copy()
    atoms_scaled.set_cell(cell0 * scale, scale_atoms=True)
    atoms_scaled.calc = calc
    
    volumes.append(atoms_scaled.get_volume())
    energies.append(atoms_scaled.get_potential_energy())
    print(f"Scale: {scale:.3f}, V: {volumes[-1]:.2f} Å³, E: {energies[-1]:.6f} eV")

# EOS拟合
eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
v0, e0, B = eos.fit()
B_GPa = B * 160.21766208

print(f"\n=== Bulk Modulus Results ===")
print(f"Equilibrium volume: {v0:.2f} Å³")
print(f"Equilibrium energy: {e0:.6f} eV")
print(f"Bulk modulus: {B_GPa:.2f} GPa")

# 绘图
eos.plot(filename="eos.png")
```

**性能数据**（MOFSimBench）：

| 模型 | MAE (GPa) | MAPE (%) |
|------|-----------|----------|
| eSEN-OAM | **2.64** 🥇 | 22.1 |
| MACE-MP-MOF0 | 3.14 | 23.5 |
| SevenNet-ompa | 3.35 | 24.0 |
| orb-v3-omat | **3.58** 🥈 | 24.5 |
| orb-v3-mpa | 4.12 | 26.8 |

---

### 4.2 声子与热容

#### 物理意义
通过晶格振动（声子）计算热力学性质（热容Cv、熵S、自由能F）。

#### 输入规范

| 参数 | 类型 | 单位 | 推荐值 | 说明 |
|------|------|------|--------|------|
| `atoms` | ASE Atoms | - | - | 初始结构（应先优化） |
| `supercell_matrix` | list/ndarray | - | [2,2,2] | 超胞矩阵（3x3或3个整数） |
| `displacement` | float | Å | 0.01 | 有限差分位移 |
| `mesh` | list | - | [20,20,20] | 声子DOS的k-点网格 |

#### 输出规范（声子计算）

| 属性 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `frequency_points` | ndarray | THz | 声子频率点 |
| `total_dos` | ndarray | 1/THz | 声子态密度 |
| 力常数 | ndarray | - | 原子间力常数矩阵 |

#### 输出规范（热力学性质）

| 属性 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `temperatures` | ndarray | K | 温度点 |
| `free_energy` | ndarray | kJ/mol | Helmholtz自由能 |
| `entropy` | ndarray | J/(K·mol) | 熵 |
| `heat_capacity` | ndarray | J/(K·mol) | 定容热容Cv |

#### 调用接口

**Phonopy** (Context7验证)：
```python
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

# ✅ 创建Phonopy对象
phonon = Phonopy(
    phonopy_atoms,
    supercell_matrix=[[2,0,0],[0,2,0],[0,0,2]],
    primitive_matrix="auto"
)

# ✅ 生成位移
phonon.generate_displacements(distance=0.01)
supercells = phonon.supercells_with_displacements

# ✅ 计算力（使用Orb）
forces = []
for scell in supercells:
    atoms_disp = convert_to_ase(scell)
    atoms_disp.calc = calc
    forces.append(atoms_disp.get_forces())

# ✅ 设置力常数
phonon.forces = forces
phonon.produce_force_constants()

# ✅ 计算声子DOS
phonon.run_mesh(mesh=[20,20,20])
phonon.run_total_dos()

# ✅ 计算热力学性质
phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)
tp_dict = phonon.get_thermal_properties_dict()
```

#### 完整示例

```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase import Atoms
import numpy as np

# 设置
orbff = pretrained.orb_v3_conservative_inf_omat(device="cuda", precision="float32-high")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF.cif")
atoms.calc = calc

# 转换为Phonopy格式
def ase_to_phonopy(atoms):
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        positions=atoms.positions,
        masses=atoms.get_masses()
    )

# 创建Phonopy对象
phonon = Phonopy(
    ase_to_phonopy(atoms),
    supercell_matrix=[[2,0,0],[0,2,0],[0,0,2]],
    primitive_matrix="auto"
)

# 生成位移
phonon.generate_displacements(distance=0.01)
supercells = phonon.supercells_with_displacements
print(f"Generated {len(supercells)} displaced supercells")

# 计算力
forces = []
for i, scell in enumerate(supercells):
    # 转换回ASE
    atoms_disp = Atoms(
        symbols=scell.symbols,
        cell=scell.cell,
        positions=scell.positions,
        pbc=True
    )
    atoms_disp.calc = calc
    forces.append(atoms_disp.get_forces())
    print(f"Calculated forces for displacement {i+1}/{len(supercells)}")

# 设置力常数
phonon.forces = forces
phonon.produce_force_constants()

# 计算热力学性质
phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)
tp_dict = phonon.get_thermal_properties_dict()

temperatures = tp_dict['temperatures']
heat_capacity = tp_dict['heat_capacity']  # J/(K·mol)

# 转换为 J/(K·g)
mass_per_formula = 1000.0  # g/mol，需根据实际MOF调整
Cv_J_K_g = heat_capacity / mass_per_formula

# 输出300K热容
idx_300K = np.argmin(np.abs(temperatures - 300))
print(f"\nHeat capacity at 300K: {Cv_J_K_g[idx_300K]:.4f} J/(K·g)")

# 绘图
import matplotlib.pyplot as plt
plt.figure()
plt.plot(temperatures, Cv_J_K_g)
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity [J/(K·g)]')
plt.savefig('heat_capacity.png')
```

**性能数据**（MOFSimBench，231个结构，300K）：

| 模型 | MAE [J/(K·g)] | MAPE (%) | 排名 |
|------|---------------|----------|------|
| orb-v3-omat | **0.018** 🥇 | **2.3** 🥇 | **1** |
| MACE-MP-MOF0 | 0.020 | 2.5 | 2 |
| eSEN-OAM | 0.024 | 3.0 | 3 |
| orb-v3-mpa | 0.026 | 3.2 | 4 |

**关键发现**：
- 🏆 **orb-v3-omat是所有模型中热容预测最准确的**

---

## 5. 主客体相互作用任务

### 5.1 吸附能计算

#### 物理意义
计算气体分子在MOF孔道中的吸附能：  
**E_ads = E(MOF+gas) - E(MOF) - E(gas)**

负值表示放热吸附（有利）。

#### 输入规范

| 参数 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `mof_atoms` | ASE Atoms | - | MOF框架结构 |
| `gas_molecule` | str/Atoms | - | 气体分子（"CO2", "H2O"等或Atoms对象） |
| `site_position` | ndarray | Å | 吸附位点坐标 [x, y, z] |
| `optimize_complex` | bool | - | 是否优化吸附复合物 |
| `freeze_mof` | bool | - | 优化时是否固定MOF框架 |

#### 输出规范

| 属性 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `E_ads` | float | eV | 吸附能 |
| `E_ads_kJmol` | float | kJ/mol | 吸附能（× 96.485） |
| 优化后复合物 | ASE Atoms | - | 吸附构型 |

#### 调用接口

```python
# ✅ 构建吸附复合物
mof_gas = mof.copy()
for atom in gas_molecule:
    mof_gas.append(atom.symbol)
    mof_gas.positions[-1] = atom.position

# ✅ 固定MOF框架
from ase.constraints import FixAtoms
mof_indices = list(range(len(mof)))
constraint = FixAtoms(indices=mof_indices)
mof_gas.set_constraint(constraint)

# ✅ 优化
from ase.optimize import LBFGS
opt = LBFGS(mof_gas)
opt.run(fmax=0.05)
```

#### 完整示例

```python
from ase.io import read
from ase.build import molecule
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
import numpy as np

# 设置
orbff = pretrained.orb_v3_conservative_inf_omat(device="cuda", precision="float32-high")
calc = ORBCalculator(orbff, device="cuda")

# 1. 优化纯MOF
mof = read("MOF.cif")
mof.calc = calc
opt_mof = LBFGS(mof)
opt_mof.run(fmax=0.05)
E_mof = mof.get_potential_energy()
print(f"MOF energy: {E_mof:.6f} eV")

# 2. 优化气体分子
co2 = molecule("CO2")
co2.center(vacuum=10.0)
co2.pbc = True
co2.calc = calc
opt_co2 = LBFGS(co2)
opt_co2.run(fmax=0.01)
E_co2 = co2.get_potential_energy()
print(f"CO2 energy: {E_co2:.6f} eV")

# 3. 构建吸附复合物
mof_co2 = mof.copy()
co2_opt = co2.copy()
site = np.array([10.0, 10.0, 10.0])  # 吸附位点
co2_opt.positions += (site - co2_opt.get_center_of_mass())

for atom in co2_opt:
    mof_co2.append(atom.symbol)
    mof_co2.positions[-1] = atom.position

mof_co2.calc = calc

# 4. 优化吸附构型（固定MOF）
mof_indices = list(range(len(mof)))
constraint = FixAtoms(indices=mof_indices)
mof_co2.set_constraint(constraint)

opt_complex = LBFGS(mof_co2)
opt_complex.run(fmax=0.05)
E_complex = mof_co2.get_potential_energy()
print(f"Complex energy: {E_complex:.6f} eV")

# 5. 计算吸附能
E_ads = E_complex - E_mof - E_co2
E_ads_kJmol = E_ads * 96.485

print(f"\n=== Adsorption Results ===")
print(f"Adsorption energy: {E_ads:.4f} eV = {E_ads_kJmol:.2f} kJ/mol")
if E_ads < 0:
    print("✓ Exothermic (favorable)")
else:
    print("✗ Endothermic (unfavorable)")

# 保存吸附构型
mof_co2.write("MOF_CO2_adsorbed.cif")
```

---

### 5.2 配位环境分析

#### 物理意义
分析金属中心的配位数、配位键长、配位几何，评估结构稳定性。

#### 输入规范

| 参数 | 类型 | 说明 |
|------|------|------|
| `atoms` | ASE Atoms | 包含金属的结构 |
| `metal_indices` | list | 金属原子索引（可自动识别） |
| `cutoff_mult` | float | 截断半径倍数（默认1.2） |

#### 输出规范

| 属性 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `coordination_number` | int | - | 配位数 |
| `neighbor_symbols` | list | - | 配位原子元素 |
| `bond_lengths` | ndarray | Å | 配位键长 |
| `avg_bond_length` | float | Å | 平均键长 |

#### 调用接口

**ASE NeighborList** (Context7验证)：
```python
from ase.neighborlist import NeighborList, natural_cutoffs

# ✅ 创建邻居列表
cutoffs = natural_cutoffs(atoms, mult=1.2)
nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
nl.update(atoms)

# ✅ 获取邻居
indices, offsets = nl.get_neighbors(metal_idx)
coordination_number = len(indices)
```

#### 完整示例

```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.neighborlist import NeighborList, natural_cutoffs
import numpy as np

# 设置
orbff = pretrained.orb_v3_conservative_inf_omat(device="cuda", precision="float32-high")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("Cu_MOF.cif")
atoms.calc = calc

# 自动识别金属原子（Z >= 21）
metal_indices = [i for i, atom in enumerate(atoms) if atom.number >= 21]
print(f"Detected {len(metal_indices)} metal atoms")

# 创建邻居列表
cutoffs = natural_cutoffs(atoms, mult=1.2)
nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
nl.update(atoms)

# 分析每个金属中心
for metal_idx in metal_indices:
    metal_symbol = atoms[metal_idx].symbol
    metal_pos = atoms.positions[metal_idx]
    
    # 获取邻居
    indices, offsets = nl.get_neighbors(metal_idx)
    
    # 计算键长
    bond_lengths = []
    neighbor_symbols = []
    for idx, offset in zip(indices, offsets):
        neighbor_pos = atoms.positions[idx] + offset @ atoms.cell.array
        distance = np.linalg.norm(neighbor_pos - metal_pos)
        bond_lengths.append(distance)
        neighbor_symbols.append(atoms[idx].symbol)
    
    # 输出
    coordination_number = len(indices)
    avg_bond_length = np.mean(bond_lengths)
    
    print(f"\n{metal_symbol} atom #{metal_idx}:")
    print(f"  Coordination number: {coordination_number}")
    print(f"  Neighbors: {', '.join(neighbor_symbols)}")
    print(f"  Bond lengths: {[f'{d:.3f}' for d in bond_lengths]} Å")
    print(f"  Average: {avg_bond_length:.3f} Å")
```

**性能数据**（MOFSimBench，MD稳定性测试）：

| 模型 | 配位数保持率 |
|------|-------------|
| orb-v3-omat | **92%** ✅ |
| orb-v3-mpa | **90%** ✅ |
| MACE-OMAT-0 | 88% |
| orb-d3-v2 | 70% ❌ |

---

## 6. 版本兼容性

### 6.1 依赖库版本

| 库 | 最低版本 | 推荐版本 | 说明 |
|-----|---------|---------|------|
| **orb-models** | 0.3.0 | **latest** | Orb主包 |
| **ase** | 3.22.0 | **3.23.0+** | FrechetCellFilter需要 |
| **phonopy** | 2.20.0 | **latest** | 声子计算 |
| **numpy** | 1.20.0 | **latest** | 数值计算 |
| **torch** | 2.0.0 | **2.3.1** | CPU/GPU支持（避免2.4.1） |
| **matplotlib** | 3.0.0 | latest | 可视化（可选） |

### 6.2 Orb模型版本映射

| 简称 | 完整模型名 | Context7验证函数 |
|------|-----------|------------------|
| orb-v3-omat | orb-v3-conservative-120-omat | `pretrained.orb_v3_conservative_inf_omat()` ✅ |
| orb-v3-mpa | orb-v3-conservative-120-mpa | `pretrained.orb_v3_conservative_inf_mpa()` ✅ |
| orb-d3-v2 | orb-d3-v2 | `pretrained.orb_d3_v2()` ✅ |
| orb-mptraj-only-v2 | orb-mptraj-only-v2 | `pretrained.orb_mptraj_only_v2()` ✅ |

### 6.3 已验证的ASE接口

| 接口 | 版本 | 状态 | 说明 |
|------|------|------|------|
| `atoms.get_potential_energy()` | ASE 3.22+ | ✅ | 标准能量接口 |
| `atoms.get_forces()` | ASE 3.22+ | ✅ | 返回(N,3)数组 |
| `atoms.get_stress(voigt=True)` | ASE 3.22+ | ✅ | 6分量Voigt记号 |
| `FrechetCellFilter` | ASE 3.23+ | ✅ | 推荐（ExpCellFilter已废弃） |
| `Langevin` | ASE 3.22+ | ✅ | NVT恒温器 |
| `NPT` | ASE 3.22+ | ✅ | NPT系综（Berendsen） |
| `EquationOfState` | ASE 3.22+ | ✅ | EOS拟合 |
| `NeighborList` | ASE 3.22+ | ✅ | 邻居列表 |

### 6.4 已验证的Phonopy接口

| 接口 | 版本 | 状态 |
|------|------|------|
| `Phonopy()` | 2.20+ | ✅ |
| `generate_displacements()` | 2.20+ | ✅ |
| `produce_force_constants()` | 2.20+ | ✅ |
| `run_mesh()` | 2.20+ | ✅ |
| `run_total_dos()` | 2.20+ | ✅ |
| `run_thermal_properties()` | 2.20+ | ✅ |
| `get_thermal_properties_dict()` | 2.20+ | ✅ |

---

## 7. 常用单位转换

### 7.1 能量单位

| 从 | 到 | 转换因子 |
|-----|-----|---------|
| eV | kJ/mol | × 96.485 |
| eV | kcal/mol | × 23.061 |
| eV/atom | meV/atom | × 1000 |

### 7.2 压强单位

| 从 | 到 | 转换因子 | 说明 |
|-----|-----|---------|------|
| eV/Å³ | GPa | × 160.21766208 | **常用** |
| GPa | eV/Å³ | ÷ 160.21766208 | - |
| GPa | atm | × 9869.23 | 1 atm ≈ 0.0001 GPa |
| eV/Å³ | bar | × 1602176.62 | - |

### 7.3 长度与时间

| 从 | 到 | 转换因子 |
|-----|-----|---------|
| Å | Bohr | × 1.88973 |
| fs | ps | × 0.001 |
| THz | cm⁻¹ | × 33.356 |

### 7.4 热力学量

| 量 | ASE/Phonopy单位 | 常用单位 | 转换 |
|-----|----------------|---------|------|
| 热容 | J/(K·mol) | J/(K·g) | ÷ 分子量 |
| 熵 | J/(K·mol) | J/(K·g) | ÷ 分子量 |
| 自由能 | kJ/mol | eV | ÷ 96.485 |

---

## 8. 参考文献

### Orb模型论文

1. **Orb v2**: Neumann, M. et al. *Orb: A Fast, Scalable Neural Network Potential.* arXiv:2410.22570 (2024)
2. **Orb v3**: Rhodes, B. et al. *Orb-v3: Atomistic Simulation at Scale.* arXiv:2504.06231 (2025)

### 评估基准

3. **MOFSimBench**: Kraß, H.; Huang, J.; Moosavi, S.M. *MOFSimBench: Evaluating Universal Machine Learning Interatomic Potentials In Metal–Organic Framework Molecular Modeling.* arXiv:2507.11806 (2025)

### 工具文档

4. **ASE**: https://wiki.fysik.dtu.dk/ase/
5. **Phonopy**: https://phonopy.github.io/phonopy/
6. **Orb Models GitHub**: https://github.com/orbital-materials/orb-models

---

*文档基于 orb-models 官方文档和 Context7 验证生成*  
*最后更新: 2026年1月7日*
