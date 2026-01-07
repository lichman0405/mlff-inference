# Orb 系列模型推理任务文档

> **模型类别**: 图网络模拟器 (Graph Network Simulator, GNS)
> 
> **文档版本**: v1.0
> 
> **最后更新**: 2026年1月7日

**相关文档**：
- [Orb 环境安装指南](INSTALL.md)：CPU/GPU 环境配置说明
- [Orb API 接口参考](Orb_inference_API_reference.md)：详细的输入输出规范和接口验证

---

## 目录

1. [模型概述](#1-模型概述)
2. [推理任务详解](#2-推理任务详解)
   - [2.1 静态建模与结构优化](#21-静态建模与结构优化)
   - [2.2 动力学建模](#22-动力学建模)
   - [2.3 体相性质预测](#23-体相性质预测)
   - [2.4 主客体相互作用](#24-主客体相互作用)
3. [基于ASE/Phonopy的可扩展推理任务](#3-基于asephonopy的可扩展推理任务)
4. [任务可行性总结](#4-任务可行性总结)

---

## 1. 模型概述

### 1.1 Orb 系列模型列表

| 模型名称 | 训练数据集 | 力类型 | 特点 | GitHub |
|----------|------------|--------|------|--------|
| **orb-d3-v2** | MPtraj + Alexandria | 非保守力 | 内置D3校正预测 | [orbital-materials/orb-models](https://github.com/orbital-materials/orb-models) |
| **orb-mptraj-only-v2** | MPtraj | 非保守力 | 仅MPtraj训练，无D3 | [orbital-materials/orb-models](https://github.com/orbital-materials/orb-models) |
| **orb-v3-con-inf-omat** | OMat24 | **保守力** | 🏆 性能最佳，无上限邻居限制 | [orbital-materials/orb-models](https://github.com/orbital-materials/orb-models) |
| **orb-v3-con-inf-mpa** | MPtraj + Alexandria | **保守力** | 更广泛化学覆盖 | [orbital-materials/orb-models](https://github.com/orbital-materials/orb-models) |

**性能亮点（基于MOFSimBench）**：
- 🥈 **综合性能第二**（仅次于eSEN-OAM）
- 🥇 **热容预测第一**（orb-v3-omat: MAE 0.018 J/K/g, MAPE 2.3%）
- 🥇 **结构优化成功率89%**（与eSEN-OAM并列）
- ✅ **MD稳定性优异**（体积漂移控制良好）
- ✅ **配位环境稳定**（金属配位数保持准确）

### 1.2 Orb 架构特点

#### 核心设计理念

**Orb 与传统等变模型的根本区别**：

| 特性 | 传统等变模型（如MACE） | Orb (GNS) |
|------|------------------------|-----------|
| **等变性实现** | 预定义（通过E(3)群表示） | **学习获得**（通过数据增强和正则化） |
| **架构复杂度** | 高（需要球谐函数、Clebsch-Gordan系数） | 低（简单的消息传递） |
| **计算效率** | 中等 | **高**（更快的训练和推理） |
| **可扩展性** | 受限于群论框架 | **灵活**（易于添加新特性） |

#### 关键技术创新

1. **学习等变性（Learned Equivariance）**
   - 不预定义对称性操作
   - 通过旋转/反射数据增强让模型自学习
   - 使用等变性损失函数正则化

2. **无上限邻居限制（Unbounded Neighbor Interactions）**
   - v3版本引入 `-inf` 后缀表示无上限
   - 特别适合MOF的大孔径结构
   - 避免截断导致的长程相互作用丢失

3. **保守力 vs 非保守力**
   - **v2版本**：直接预测力（非保守），计算快但MD不稳定
   - **v3版本**：通过能量梯度计算力（保守），稳定性大幅提升

4. **工业级验证**
   - Orbital Materials公司背景
   - 在实际材料发现项目中验证
   - 强调可靠性和鲁棒性

### 1.3 安装与基本使用

详细安装说明请参考 [INSTALL.md](INSTALL.md)。

**快速安装（CPU版本）**：
```bash
# 创建环境
conda create -n orb-cpu python=3.10
conda activate orb-cpu

# 安装核心依赖
pip install orb-models ase phonopy numpy

# 验证安装
python -c "import orb_models; print('Orb models installed!')"
```

**基本使用示例**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

# 加载预训练模型（orb-v3-omat推荐用于MOF）
orbff = pretrained.orb_v3()
calc = ORBCalculator(orbff, device="cpu")

# 读取结构
atoms = read("structure.cif")
atoms.calc = calc

# 计算能量和力
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Å
print(f"Energy: {energy:.4f} eV")
```

**模型选择指南**：

```python
# 根据应用场景选择模型
from orb_models.forcefield import pretrained

# 方案1: MOF推理任务 → orb-v3-omat（推荐）
orbff = pretrained.orb_v3(model="omat-v3")

# 方案2: 更广化学覆盖 → orb-v3-mpa
orbff = pretrained.orb_v3(model="mpa-v3")

# 方案3: 快速计算（牺牲稳定性）→ orb-v2-d3
orbff = pretrained.orb_d3_v2()

# 方案4: 无D3校正 → orb-mptraj-only-v2
orbff = pretrained.orb_mptraj_only_v2()
```

---

## 2. 推理任务详解

### 2.1 静态建模与结构优化

#### 2.1.1 单点能量计算

**物理意义**：计算给定原子构型的势能、原子受力和应力张量。

**代码示例**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
import numpy as np

# 加载模型和结构
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")  # 使用GPU加速

atoms = read("MOF.cif")
atoms.calc = calc

# 单点能量计算
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress(voigt=True)  # 6分量Voigt记号

# 计算派生量
energy_per_atom = energy / len(atoms)
max_force = np.max(np.linalg.norm(forces, axis=1))
rms_force = np.sqrt(np.mean(np.sum(forces**2, axis=1)))
pressure_GPa = -np.trace(stress[:3]) / 3 * 160.21766208  # eV/Å³ → GPa

print(f"Total Energy: {energy:.6f} eV")
print(f"Energy per atom: {energy_per_atom:.6f} eV/atom")
print(f"Max Force: {max_force:.6f} eV/Å")
print(f"RMS Force: {rms_force:.6f} eV/Å")
print(f"Pressure: {pressure_GPa:.4f} GPa")
```

**性能提示**：
- ✅ Orb在CPU上比MACE快约1.5-2倍
- ✅ GPU加速更显著（10-50倍）
- ⚠️ 大体系（>1000原子）建议使用GPU

---

#### 2.1.2 结构优化

**物理意义**：优化原子位置和晶胞参数，使系统达到势能最小。

**代码示例（原子位置优化）**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.optimize import LBFGS

# 设置
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF_initial.cif")
atoms.calc = calc

# 结构优化
optimizer = LBFGS(atoms, trajectory="opt.traj", logfile="opt.log")
optimizer.run(fmax=0.05, steps=500)

# 保存优化后结构
atoms.write("MOF_optimized.cif")

print(f"Final Energy: {atoms.get_potential_energy():.6f} eV")
print(f"Optimization converged in {optimizer.nsteps} steps")
```

**代码示例（晶胞+原子联合优化）**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.optimize import LBFGS
from ase.constraints import FrechetCellFilter

# 使用FrechetCellFilter允许晶胞变化（ASE >= 3.23.0推荐）
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF_initial.cif")
atoms.calc = calc

# 创建FrechetCellFilter（同时优化晶胞和原子）
ecf = FrechetCellFilter(atoms)
optimizer = LBFGS(ecf, trajectory="opt_cell.traj", logfile="opt_cell.log")
optimizer.run(fmax=0.05, steps=500)

# 保存结果
atoms.write("MOF_optimized_cell.cif")

# 输出晶胞变化
print("Lattice parameter change:")
print(f"a: {atoms.cell.lengths()[0]:.4f} Å")
print(f"b: {atoms.cell.lengths()[1]:.4f} Å")
print(f"c: {atoms.cell.lengths()[2]:.4f} Å")
```

**收敛性对比（基于MOFSimBench）**：

| 模型 | 收敛成功率 | 平均步数 | 体积偏差<10% |
|------|-----------|---------|-------------|
| orb-v3-omat | **89%** 🥇 | 120 | ✅ |
| orb-v3-mpa | 87% | 125 | ✅ |
| orb-d3-v2 | 61% ❌ | 200+ | ❌ |
| orb-mptraj-only-v2 | 65% | 180+ | ❌ |

**关键发现**：
- ✅ **v3保守力模型收敛性远优于v2非保守力**
- ✅ orb-v3-omat与eSEN-OAM并列最佳（89%）
- ⚠️ 避免使用v2模型做结构优化

---

### 2.2 动力学建模

#### 2.2.1 NVT分子动力学（恒温恒容）

**物理意义**：固定温度和体积，研究MOF框架的热稳定性、客体分子扩散等。

**代码示例（NVT MD稳定性测试）**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# 设置
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF.cif")
atoms.calc = calc

# 初始化速度（300K）
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# NVT MD（Langevin恒温器）
timestep = 1.0 * units.fs
temperature_K = 300
friction = 0.01  # 1/fs，或使用taut=1/friction=100fs

dyn = Langevin(
    atoms,
    timestep=timestep,
    temperature_K=temperature_K,
    friction=friction,
    trajectory="nvt_md.traj",
    logfile="nvt_md.log",
    loginterval=100
)

# 运行50ps
dyn.run(steps=50000)

print("NVT MD completed: 50 ps simulation")
```

**代码示例（客体分子扩散系数计算）**：
```python
from ase.io import read, Trajectory
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
import numpy as np

# 设置（MOF吸附CO2体系）
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF_CO2.cif")
atoms.calc = calc

# 初始化并运行NVT MD
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(
    atoms, 
    timestep=1.0*units.fs, 
    temperature_K=300, 
    friction=0.01,
    trajectory="diffusion.traj"
)
dyn.run(steps=100000)  # 100 ps

# 分析扩散系数（只分析CO2分子）
traj = Trajectory("diffusion.traj")
co2_indices = [i for i in range(len(atoms)) if atoms[i].symbol == 'C' and i < 100]  # 示例

# 计算均方位移（MSD）
positions = []
for frame in traj:
    positions.append(frame.positions[co2_indices])
positions = np.array(positions)

# MSD = <|r(t) - r(0)|²>
msd = np.mean(np.sum((positions - positions[0])**2, axis=2), axis=1)
time = np.arange(len(msd)) * 1.0  # fs

# 线性拟合提取扩散系数：MSD = 6Dt
from scipy.stats import linregress
slope, _, _, _, _ = linregress(time[1000:], msd[1000:])  # 忽略初始不稳定部分
D = slope / 6  # Å²/fs
D_cm2_s = D * 1e-16 / 1e-15  # 转换为 cm²/s

print(f"Diffusion coefficient: {D:.6f} Å²/fs = {D_cm2_s:.2e} cm²/s")
```

**性能提示（基于MOFSimBench）**：
- ✅ orb-v3系列：体积漂移 < 5%（50ps@300K）
- ❌ orb-v2系列：体积漂移 > 20%（非保守力导致能量漂移）

---

#### 2.2.2 NPT分子动力学（恒温恒压）

**物理意义**：固定温度和压力，允许晶胞变化，模拟真实实验条件。

**代码示例**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# 设置
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF.cif")
atoms.calc = calc

# 初始化速度
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# NPT MD（Berendsen barostat）
timestep = 1.0 * units.fs
temperature_K = 300
pressure_GPa = 0.0  # 1 atm ≈ 0.0001 GPa

# 估算pfactor（基于体积模量）
# pfactor = (timestep^2) * B / V，B是体积模量
# 对于MOF，B ~ 10-30 GPa，可以使用默认值或手动设置
volume = atoms.get_volume()
bulk_modulus_GPa = 20.0  # 估算值
pfactor = (timestep**2) * bulk_modulus_GPa / volume / 160.21766208  # 单位转换

dyn = NPT(
    atoms,
    timestep=timestep,
    temperature_K=temperature_K,
    externalstress=pressure_GPa / 160.21766208,  # GPa → eV/Å³
    ttime=100*units.fs,  # 温度弛豫时间
    pfactor=pfactor,
    trajectory="npt_md.traj",
    logfile="npt_md.log",
    loginterval=100
)

# 运行50ps
dyn.run(steps=50000)

# 分析体积变化
from ase.io import Trajectory
traj = Trajectory("npt_md.traj")
volumes = [frame.get_volume() for frame in traj]
import numpy as np
print(f"Initial volume: {volumes[0]:.2f} Å³")
print(f"Final volume: {volumes[-1]:.2f} Å³")
print(f"Volume drift: {(volumes[-1]/volumes[0] - 1)*100:.2f}%")
print(f"Average volume: {np.mean(volumes):.2f} ± {np.std(volumes):.2f} Å³")
```

**配位环境稳定性测试（基于MOFSimBench方案）**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.neighborlist import NeighborList, natural_cutoffs

def get_coordination_numbers(atoms, metal_indices):
    """计算金属原子的配位数"""
    cutoffs = natural_cutoffs(atoms, mult=1.2)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    coord_numbers = []
    for metal_idx in metal_indices:
        indices, offsets = nl.get_neighbors(metal_idx)
        coord_numbers.append(len(indices))
    return coord_numbers

# 设置
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("Cu_MOF.cif")
atoms.calc = calc

# 识别Cu原子
metal_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'Cu']
initial_coord = get_coordination_numbers(atoms, metal_indices)
print(f"Initial Cu coordination: {initial_coord}")

# 温度循环测试：300K → 400K → 300K
for temp in [300, 400, 300]:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    dyn = NPT(
        atoms, 
        timestep=1.0*units.fs, 
        temperature_K=temp,
        externalstress=0.0,
        ttime=100*units.fs,
        pfactor=None  # 自动估算
    )
    dyn.run(steps=10000)  # 10ps per stage

final_coord = get_coordination_numbers(atoms, metal_indices)
print(f"Final Cu coordination: {final_coord}")
print(f"Coordination preserved: {initial_coord == final_coord}")
```

**性能对比（MOFSimBench，13个Cu-MOF）**：

| 模型 | 配位数保持率 | 平均偏差 |
|------|-------------|---------|
| orb-v3-omat | 92% ✅ | 0.15 |
| orb-v3-mpa | 90% ✅ | 0.18 |
| orb-d3-v2 | 70% ❌ | 0.45 |

---

### 2.3 体相性质预测

#### 2.3.1 体积模量计算（Bulk Modulus）

**物理意义**：材料抵抗均匀压缩的能力，B₀ = -V(∂P/∂V)。

**代码示例（Birch-Murnaghan EOS拟合）**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.eos import EquationOfState
import numpy as np

# 设置
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("MOF.cif")
atoms.calc = calc

# 生成体积缩放点（±4%，11个点）
volumes = []
energies = []
cell0 = atoms.cell.copy()
volume0 = atoms.get_volume()

for scale in np.linspace(0.96, 1.04, 11):
    atoms_scaled = atoms.copy()
    atoms_scaled.set_cell(cell0 * scale, scale_atoms=True)
    atoms_scaled.calc = calc
    
    volumes.append(atoms_scaled.get_volume())
    energies.append(atoms_scaled.get_potential_energy())

# EOS拟合
eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
v0, e0, B = eos.fit()

# 单位转换：eV/Å³ → GPa
B_GPa = B * 160.21766208

print(f"Equilibrium volume: {v0:.2f} Å³")
print(f"Equilibrium energy: {e0:.6f} eV")
print(f"Bulk modulus: {B_GPa:.2f} GPa")

# 可视化EOS曲线
eos.plot(filename="eos.png")
```

**性能评估（基于MOFSimBench）**：

| 模型 | MAE (GPa) | MAPE (%) | 系统性偏差 |
|------|-----------|----------|-----------|
| orb-v3-omat | **3.58** 🥈 | 24.5 | 轻微低估 |
| orb-v3-mpa | 4.12 | 26.8 | 轻微低估 |
| eSEN-OAM | **2.64** 🥇 | 22.1 | - |
| orb-d3-v2 | 72.29 ❌ | 450+ | 严重高估 |

**关键发现**：
- ✅ orb-v3系列表现优秀，仅次于eSEN和MACE-MOF0
- ❌ orb-v2非保守力导致体积模量预测失败
- ⚠️ 所有模型存在轻微低估（与势能面软化相关）

---

#### 2.3.2 声子计算与热容

**物理意义**：通过晶格振动（声子）计算热力学性质（热容、熵、自由能）。

**代码示例（声子计算）**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np

# 设置
orbff = pretrained.orb_v3(model="omat-v3")
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

# 创建超胞（2x2x2）
supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
phonon = Phonopy(
    ase_to_phonopy(atoms),
    supercell_matrix=supercell_matrix,
    primitive_matrix="auto"
)

# 生成位移
phonon.generate_displacements(distance=0.01)  # Å
supercells = phonon.supercells_with_displacements

# 计算力（使用Orb）
forces = []
for scell in supercells:
    # 转换回ASE
    from ase import Atoms
    atoms_disp = Atoms(
        symbols=scell.symbols,
        cell=scell.cell,
        positions=scell.positions,
        pbc=True
    )
    atoms_disp.calc = calc
    forces.append(atoms_disp.get_forces())

# 设置力常数
phonon.forces = forces
phonon.produce_force_constants()

# 计算声子DOS
phonon.run_mesh(mesh=[20, 20, 20])
phonon.run_total_dos()
dos_dict = phonon.get_total_dos_dict()

# 绘制声子DOS
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(dos_dict['frequency_points'], dos_dict['total_dos'])
plt.xlabel('Frequency (THz)')
plt.ylabel('DOS')
plt.title('Phonon Density of States')
plt.savefig('phonon_dos.png', dpi=300)
plt.close()

print("Phonon DOS saved to phonon_dos.png")
```

**代码示例（热力学性质）**：
```python
# 接上面的代码，在计算完力常数后

# 计算热力学性质（0-1000K）
phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)
tp_dict = phonon.get_thermal_properties_dict()

temperatures = tp_dict['temperatures']      # K
free_energy = tp_dict['free_energy']        # kJ/mol
entropy = tp_dict['entropy']                # J/K/mol
heat_capacity = tp_dict['heat_capacity']    # J/K/mol

# 转换为单位质量（假设MOF总质量1000 g/mol）
mass_per_formula = 1000.0  # g/mol，需根据实际MOF调整
Cv_J_K_g = heat_capacity / mass_per_formula

# 输出300K的热容
idx_300K = np.argmin(np.abs(temperatures - 300))
print(f"Heat capacity at 300K: {Cv_J_K_g[idx_300K]:.4f} J/K/g")

# 绘制热容曲线
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(temperatures, Cv_J_K_g)
plt.xlabel('Temperature (K)')
plt.ylabel('Heat Capacity (J/K/g)')
plt.title('Heat Capacity vs Temperature')
plt.grid(alpha=0.3)
plt.savefig('heat_capacity.png', dpi=300)
plt.close()

print("Heat capacity curve saved to heat_capacity.png")
```

**性能评估（MOFSimBench，231个结构，300K热容）**：

| 模型 | MAE (J/K/g) | MAPE (%) | 排名 |
|------|-------------|----------|------|
| orb-v3-omat | **0.018** 🥇 | **2.3** 🥇 | 1 |
| MACE-MP-MOF0 | 0.020 | 2.5 | 2 |
| eSEN-OAM | 0.024 | 3.0 | 3 |
| orb-v3-mpa | 0.026 | 3.2 | 4 |
| orb-d3-v2 | 0.055 ❌ | 6.8 | - |

**关键发现**：
- 🏆 **orb-v3-omat是所有模型中热容预测最准确的**
- ✅ 显著优于其他通用模型
- ⚠️ 所有模型存在系统性高估（势能面软化问题）

**性能提示**：
- ⚠️ 声子计算在CPU上较慢（建议GPU）
- ⚠️ 大超胞（>500原子）计算时间可达数小时
- ✅ 可以使用更小的位移（0.005Å）提高精度

---

### 2.4 主客体相互作用

#### 2.4.1 气体吸附能计算

**物理意义**：计算气体分子在MOF孔道中的吸附能，E_ads = E(MOF+gas) - E(MOF) - E(gas)。

**代码示例（CO₂吸附）**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
import numpy as np

# 设置
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")

# 1. 优化纯MOF
mof = read("MOF.cif")
mof.calc = calc
opt_mof = LBFGS(mof, trajectory="mof_opt.traj")
opt_mof.run(fmax=0.05)
E_mof = mof.get_potential_energy()
print(f"MOF energy: {E_mof:.6f} eV")

# 2. 优化气体分子（真空中）
co2 = molecule("CO2")
co2.center(vacuum=10.0)  # 10Å真空层
co2.pbc = True
co2.calc = calc
opt_co2 = LBFGS(co2)
opt_co2.run(fmax=0.01)
E_co2 = co2.get_potential_energy()
print(f"CO2 energy: {E_co2:.6f} eV")

# 3. 构建吸附复合物
mof_co2 = mof.copy()
# 在MOF孔道中心放置CO2（需根据实际结构调整位置）
co2_center = np.array([10.0, 10.0, 10.0])  # 示例位置
co2_optimized = co2.copy()
co2_optimized.positions += (co2_center - co2_optimized.get_center_of_mass())

# 添加CO2到MOF
for atom in co2_optimized:
    mof_co2.append(atom.symbol)
    mof_co2.positions[-1] = atom.position

mof_co2.calc = calc

# 4. 优化吸附构型（固定MOF框架，仅优化CO2）
mof_indices = list(range(len(mof)))
constraint = FixAtoms(indices=mof_indices)
mof_co2.set_constraint(constraint)

opt_complex = LBFGS(mof_co2, trajectory="complex_opt.traj")
opt_complex.run(fmax=0.05)
E_complex = mof_co2.get_potential_energy()
print(f"Complex energy: {E_complex:.6f} eV")

# 5. 计算吸附能
E_ads = E_complex - E_mof - E_co2
print(f"\nAdsorption energy: {E_ads:.4f} eV = {E_ads*96.485:.2f} kJ/mol")

# 负值表示吸附是放热过程
if E_ads < 0:
    print("✓ Exothermic adsorption (favorable)")
else:
    print("✗ Endothermic adsorption (unfavorable)")
```

**代码示例（多种气体吸附对比）**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.build import molecule
from ase.optimize import LBFGS
import numpy as np

# 设置
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")

# 优化MOF
mof = read("MOF.cif")
mof.calc = calc
opt_mof = LBFGS(mof)
opt_mof.run(fmax=0.05)
E_mof = mof.get_potential_energy()

# 测试多种气体
gas_molecules = {
    'H2O': 'H2O',
    'CO2': 'CO2',
    'CH4': 'CH4',
    'N2': 'N2',
    'H2': 'H2'
}

results = {}
for name, formula in gas_molecules.items():
    # 优化气体分子
    gas = molecule(formula)
    gas.center(vacuum=10.0)
    gas.pbc = True
    gas.calc = calc
    opt_gas = LBFGS(gas)
    opt_gas.run(fmax=0.01)
    E_gas = gas.get_potential_energy()
    
    # 构建并优化吸附复合物（简化版，实际需要多位点采样）
    mof_gas = mof.copy()
    gas_opt = gas.copy()
    # 放置在孔道中心（示例位置）
    site = np.array([10.0, 10.0, 10.0])
    gas_opt.positions += (site - gas_opt.get_center_of_mass())
    
    for atom in gas_opt:
        mof_gas.append(atom.symbol)
        mof_gas.positions[-1] = atom.position
    
    mof_gas.calc = calc
    # 简化：不固定MOF，完全优化
    opt_complex = LBFGS(mof_gas)
    opt_complex.run(fmax=0.05)
    E_complex = mof_gas.get_potential_energy()
    
    # 计算吸附能
    E_ads = E_complex - E_mof - E_gas
    results[name] = E_ads
    print(f"{name}: {E_ads:.4f} eV ({E_ads*96.485:.2f} kJ/mol)")

# 排序（从最强吸附到最弱）
sorted_results = sorted(results.items(), key=lambda x: x[1])
print("\n=== Adsorption Strength Ranking ===")
for i, (name, E_ads) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {E_ads:.4f} eV")
```

**性能评估（GoldDAC数据集）**：

orb-v3模型在主客体相互作用能预测上表现优异：

| 相互作用区域 | Orb-v3-omat表现 |
|-------------|----------------|
| 排斥区 (R) | 良好 |
| 平衡区 (E) | **优异** ✅ |
| 弱吸引区 (W) | 良好 |

**相比其他模型**：
- ✅ 优于大部分通用模型
- ≈ 与MatterSim、eSEN-OAM相当
- ✅ 优于微调模型MACE-DAC-1（在某些区域）

---

#### 2.4.2 配位环境分析

**物理意义**：分析金属中心的配位数、配位键长、配位几何。

**代码示例**：
```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from ase.neighborlist import NeighborList, natural_cutoffs
import numpy as np

# 设置
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cuda")

atoms = read("Cu_MOF.cif")
atoms.calc = calc

# 自动识别金属原子（原子序数>=21）
metal_indices = [i for i, atom in enumerate(atoms) if atom.number >= 21]
print(f"Detected {len(metal_indices)} metal atoms: {[atoms[i].symbol for i in metal_indices]}")

# 创建邻居列表（使用自然截断半径的1.2倍）
cutoffs = natural_cutoffs(atoms, mult=1.2)
nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
nl.update(atoms)

# 分析每个金属中心
coordination_info = []
for metal_idx in metal_indices:
    metal_symbol = atoms[metal_idx].symbol
    metal_pos = atoms.positions[metal_idx]
    
    # 获取邻居
    indices, offsets = nl.get_neighbors(metal_idx)
    
    # 计算配位键长
    bond_lengths = []
    neighbor_symbols = []
    for idx, offset in zip(indices, offsets):
        neighbor_pos = atoms.positions[idx] + offset @ atoms.cell.array
        distance = np.linalg.norm(neighbor_pos - metal_pos)
        bond_lengths.append(distance)
        neighbor_symbols.append(atoms[idx].symbol)
    
    # 统计配位信息
    coordination_number = len(indices)
    avg_bond_length = np.mean(bond_lengths) if bond_lengths else 0.0
    
    info = {
        'metal_index': metal_idx,
        'metal_symbol': metal_symbol,
        'coordination_number': coordination_number,
        'neighbor_symbols': neighbor_symbols,
        'bond_lengths': bond_lengths,
        'avg_bond_length': avg_bond_length
    }
    coordination_info.append(info)
    
    # 输出
    print(f"\n{metal_symbol} atom #{metal_idx}:")
    print(f"  Coordination number: {coordination_number}")
    print(f"  Neighbors: {', '.join(neighbor_symbols)}")
    print(f"  Bond lengths: {[f'{d:.3f}' for d in bond_lengths]} Å")
    print(f"  Average bond length: {avg_bond_length:.3f} Å")

# 统计所有金属的配位数分布
from collections import Counter
coord_distribution = Counter([info['coordination_number'] for info in coordination_info])
print("\n=== Coordination Number Distribution ===")
for cn, count in sorted(coord_distribution.items()):
    print(f"CN={cn}: {count} atoms")
```

**配位环境稳定性评估（MD后对比）**：
```python
# 在上面代码基础上，运行MD后重新分析

from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

# 记录初始配位数
initial_coord_numbers = [info['coordination_number'] for info in coordination_info]

# 运行高温MD测试（300K → 400K → 300K）
for temp in [300, 400, 300]:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp)
    dyn = NPT(atoms, timestep=1.0*units.fs, temperature_K=temp, 
              externalstress=0.0, ttime=100*units.fs)
    dyn.run(steps=10000)  # 10ps

# 重新分析配位环境
nl.update(atoms)
final_coord_numbers = []
for metal_idx in metal_indices:
    indices, _ = nl.get_neighbors(metal_idx)
    final_coord_numbers.append(len(indices))

# 对比
print("\n=== Coordination Stability Test ===")
print("Metal | Initial CN | Final CN | Change")
print("-" * 45)
for i, metal_idx in enumerate(metal_indices):
    initial_cn = initial_coord_numbers[i]
    final_cn = final_coord_numbers[i]
    change = final_cn - initial_cn
    symbol = atoms[metal_idx].symbol
    status = "✓" if change == 0 else "✗"
    print(f"{symbol:5} | {initial_cn:10} | {final_cn:8} | {change:+6} {status}")

# 统计稳定性
unchanged = sum(1 for i, f in zip(initial_coord_numbers, final_coord_numbers) if i == f)
stability_rate = unchanged / len(metal_indices) * 100
print(f"\nCoordination stability: {stability_rate:.1f}% ({unchanged}/{len(metal_indices)})")
```

**性能评估（MOFSimBench，13个Cu-MOF）**：

| 模型 | 配位数保持率 | 备注 |
|------|-------------|------|
| orb-v3-omat | **92%** ✅ | 优异 |
| orb-v3-mpa | **90%** ✅ | 优异 |
| MACE-OMAT-0 | 88% | 良好 |
| orb-d3-v2 | 70% ❌ | 非保守力不稳定 |

---

## 3. 基于ASE/Phonopy的可扩展推理任务

以下任务可使用相同工具链（ASE、Phonopy）和Orb模型执行：

### 3.1 热力学性质

| 任务 | 方法 | Orb优势 |
|------|------|---------|
| **声子谱** | Phonopy | ✅ 热容预测最佳 |
| **热导率** | Phono3py / Green-Kubo | ✅ 稳定MD支持 |
| **自由能** | 热力学积分 | ✅ 保守力保证 |
| **熵** | 声子方法 | ✅ 高精度声子 |
| **热膨胀系数** | 准谐近似 / NPT MD | ✅ NPT稳定性好 |

### 3.2 力学性质

| 任务 | 方法 | Orb优势 |
|------|------|---------|
| **弹性张量** | 应力-应变分析 | ✅ 应力计算准确 |
| **杨氏模量** | 从弹性张量导出 | ✅ 体积模量优异 |
| **泊松比** | 弹性常数 | ✅ 完整力学描述 |

### 3.3 动力学性质

| 任务 | 方法 | Orb优势 |
|------|------|---------|
| **扩散系数** | MSD分析 | ✅ 长时间MD稳定 |
| **粘度** | NEMD | ✅ 能量守恒好 |
| **离子电导率** | 电流自相关 | ✅ 适合电化学 |

### 3.4 吸附性质

| 任务 | 方法 | Orb优势 |
|------|------|---------|
| **吸附等温线** | GCMC (需RASPA) | ✅ 准确相互作用 |
| **等量吸附热** | Widom插入 | ✅ 能量计算快 |
| **选择性** | 多组分GCMC | ✅ 多分子体系 |
| **Henry常数** | 低压极限 | ✅ 适合筛选 |

### 3.5 特殊性质

| 任务 | 方法 | Orb优势 |
|------|------|---------|
| **框架柔性** | 变压MD | ✅ 晶胞变化稳定 |
| **负热膨胀** | 准谐近似 | ✅ 声子计算准 |
| **相变** | 自由能计算 | ✅ 热力学一致性 |

---

## 4. 任务可行性总结

### 4.1 Orb模型推荐用途

| 任务类别 | 推荐模型 | 优先级 | 备注 |
|----------|----------|--------|------|
| **MOF通用推理** | orb-v3-omat | ⭐⭐⭐⭐⭐ | 综合性能第二 |
| **热容预测** | orb-v3-omat | ⭐⭐⭐⭐⭐ | **最准确** 🥇 |
| **长时间MD** | orb-v3系列 | ⭐⭐⭐⭐⭐ | 稳定性优异 |
| **结构优化** | orb-v3系列 | ⭐⭐⭐⭐⭐ | 89%成功率 |
| **吸附模拟** | orb-v3-omat | ⭐⭐⭐⭐ | 相互作用准确 |
| **体积模量** | orb-v3-omat | ⭐⭐⭐⭐ | MAE 3.58 GPa |
| **快速计算（牺牲精度）** | orb-v2系列 | ⭐⭐ | ⚠️ 非保守力 |

### 4.2 与MACE对比

| 特性 | Orb (v3) | MACE (OMAT-0) |
|------|----------|---------------|
| **综合性能** | 🥈 第二 | Top 5 |
| **热容预测** | 🥇 **第一** | 良好 |
| **结构优化** | 🥇 89% | 良好 |
| **计算速度** | ✅ **更快** | 中等 |
| **架构复杂度** | ✅ **简单** | 复杂 |
| **大体系** | ✅ **无邻居限制** | 有限制 |
| **可解释性** | 中等 | ✅ 理论基础强 |

### 4.3 版本选择决策树

```
需要高精度预测？
├─ 是 → 使用 orb-v3 系列
│   ├─ MOF应用 → orb-v3-omat ⭐⭐⭐⭐⭐
│   └─ 广泛材料 → orb-v3-mpa
└─ 否，只需快速估算 → 使用 orb-v2 系列
    ├─ 需要D3校正 → orb-d3-v2
    └─ 不需要D3 → orb-mptraj-only-v2

需要长时间MD（>10ps）？
└─ 必须使用 orb-v3（保守力）⚠️

需要结构优化？
└─ 强烈推荐 orb-v3 ⚠️

需要热力学性质？
└─ 首选 orb-v3-omat 🥇
```

### 4.4 限制与注意事项

| 限制 | 说明 | 解决方案 |
|------|------|---------|
| **元素覆盖** | 训练数据覆盖的元素 | 检查模型支持的元素列表 |
| **v2稳定性** | 非保守力导致MD不稳定 | ⚠️ 仅用于单点计算 |
| **声子计算** | CPU上较慢 | ✅ 使用GPU加速 |
| **超大体系** | >2000原子可能内存不足 | 使用更大GPU或批处理 |

---

## 5. 最佳实践建议

### 5.1 设备选择

```python
# CPU: 适合小体系（<500原子）或快速测试
orbff = pretrained.orb_v3(model="omat-v3")
calc = ORBCalculator(orbff, device="cpu")

# GPU: 推荐用于所有生产计算
calc = ORBCalculator(orbff, device="cuda")  # 单GPU
calc = ORBCalculator(orbff, device="cuda:0")  # 指定GPU
```

### 5.2 模型加载缓存

```python
# 全局加载模型，避免重复初始化
orbff = pretrained.orb_v3(model="omat-v3")

# 在循环中复用
for structure_file in structure_list:
    atoms = read(structure_file)
    calc = ORBCalculator(orbff, device="cuda")  # 轻量级计算器
    atoms.calc = calc
    # ... 计算
```

### 5.3 批处理优化

```python
# 对于大规模筛选，使用批处理
import concurrent.futures

def calculate_energy(structure_file, orbff):
    atoms = read(structure_file)
    calc = ORBCalculator(orbff, device="cuda")
    atoms.calc = calc
    return atoms.get_potential_energy()

orbff = pretrained.orb_v3(model="omat-v3")
structure_files = ["MOF1.cif", "MOF2.cif", ...]

# 并行计算（GPU需注意内存）
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    energies = list(executor.map(lambda f: calculate_energy(f, orbff), structure_files))
```

### 5.4 错误处理

```python
from ase.io import read
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

try:
    orbff = pretrained.orb_v3(model="omat-v3")
    calc = ORBCalculator(orbff, device="cuda")
    atoms = read("MOF.cif")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
except Exception as e:
    print(f"Calculation failed: {e}")
    # 降级到CPU
    calc = ORBCalculator(orbff, device="cpu")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
```

---

## 6. 引用与参考

### 论文引用

如果在研究中使用Orb模型，请引用：

**Orb v2**:
```bibtex
@article{neumann2024orb,
  title={Orb: A Fast, Scalable Neural Network Potential},
  author={Neumann, Mark and others},
  journal={arXiv preprint arXiv:2410.22570},
  year={2024}
}
```

**Orb v3**:
```bibtex
@article{rhodes2025orb,
  title={Orb-v3: Atomistic Simulation at Scale},
  author={Rhodes, Benjamin and others},
  journal={arXiv preprint arXiv:2504.06231},
  year={2025}
}
```

**MOFSimBench评估**:
```bibtex
@article{krass2025mofsimbench,
  title={MOFSimBench: Evaluating Universal Machine Learning Interatomic Potentials In Metal–Organic Framework Molecular Modeling},
  author={Kraß, Hendrik and Huang, Ju and Moosavi, Seyed Mohamad},
  journal={arXiv preprint arXiv:2507.11806},
  year={2025}
}
```

### 相关链接

- **GitHub**: https://github.com/orbital-materials/orb-models
- **文档**: https://docs.orbitalmaterials.com/
- **Orbital Materials**: https://www.orbitalmaterials.com/
- **Hugging Face**: https://huggingface.co/orbital-materials

---

*文档生成时间: 2026年1月7日*

*基于 Orb v3 和 MOFSimBench 论文整理*
