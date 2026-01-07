# eSEN Inference - API 参考文档

> **eSEN (Smooth & Expressive Equivariant Networks)**: MOFSimBench 排名 **#1** 的通用机器学习力场  
> **开发团队**: Meta FAIR - Fu et al. 2025  
> **模型来源**: [FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem)  
> **论文**: arXiv:2502.12147

---

## 目录

1. [核心类 - ESENInference](#核心类---eseninference)
2. [单点能量计算](#单点能量计算)
3. [结构优化](#结构优化)
4. [分子动力学](#分子动力学)
5. [声子计算](#声子计算)
6. [力学性质](#力学性质)
7. [吸附能计算](#吸附能计算)
8. [配位分析](#配位分析)
9. [工具函数](#工具函数)
10. [设备管理](#设备管理)

---

## 核心类 - ESENInference

### 类定义

```python
class ESENInference:
    """eSEN 推理引擎 - MOFSimBench #1 模型
    
    基于 FAIR-Chem fairchem 框架的 eSEN (Smooth & Expressive Equivariant Networks) 模型。
    支持 8 大推理任务：能量、优化、MD、声子、力学、吸附、配位、高通量筛选。
    
    核心优势:
    - 能量预测精度 #1: MAE 0.041 eV/atom
    - 体积模量精度 #1: MAE 2.64 GPa
    - MD 稳定性 #1: 与 MatterSim 并列
    - 结构优化成功率 #1: 89% (与 orb-v3-omat 并列)
    """
```

### 初始化

```python
def __init__(
    self,
    model_name: str = 'esen-30m-oam',
    device: str = 'cuda',
    precision: str = 'float32',
    checkpoint_path: Optional[str] = None,
    cpu_threads: Optional[int] = None
)
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | str | `'esen-30m-oam'` | 模型名称：`'esen-30m-oam'` (推荐) 或 `'esen-30m-mp'` |
| `device` | str | `'cuda'` | 计算设备：`'cuda'`, `'cpu'`, `'cuda:0'` 等 |
| `precision` | str | `'float32'` | 计算精度：`'float32'` (默认) 或 `'float64'` |
| `checkpoint_path` | str | `None` | 自定义检查点路径（可选） |
| `cpu_threads` | int | `None` | CPU 线程数（仅 CPU 模式） |

#### 可用模型

| 模型名称 | 训练数据 | 参数量 | 推荐用途 |
|----------|----------|--------|----------|
| `'esen-30m-oam'` | OMat24 + MPtraj + sAlex | 30M | **通用 MOF 建模 (强烈推荐)** |
| `'esen-30m-mp'` | MPtraj only | 30M | Materials Project 数据专用 |

#### 返回

- **`ESENInference`**: 推理引擎实例

#### 示例

```python
from esen_inference import ESENInference

# 标准初始化 (GPU, float32)
esen = ESENInference(
    model_name='esen-30m-oam',
    device='cuda',
    precision='float32'
)

# 高精度模式 (GPU, float64)
esen_hp = ESENInference(
    model_name='esen-30m-oam',
    device='cuda',
    precision='float64'
)

# CPU 模式 (多线程)
esen_cpu = ESENInference(
    model_name='esen-30m-oam',
    device='cpu',
    cpu_threads=16
)

# Materials Project 专用模型
esen_mp = ESENInference(
    model_name='esen-30m-mp',
    device='cuda'
)

# 自定义检查点
esen_custom = ESENInference(
    checkpoint_path='/path/to/checkpoint.pt',
    device='cuda'
)
```

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `calculator` | `OCPCalculator` | ASE 计算器对象 (FAIR-Chem) |
| `model` | `torch.nn.Module` | eSEN 模型 |
| `device` | `torch.device` | 计算设备 |
| `precision` | `torch.dtype` | 数值精度 |
| `model_name` | `str` | 模型名称 |

---

## 单点能量计算

### `single_point()`

计算给定结构的能量、力和应力。

```python
def single_point(
    atoms: Atoms,
    properties: List[str] = ['energy', 'forces', 'stress']
) -> Dict[str, Any]
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `atoms` | `ase.Atoms` | - | 原子结构 |
| `properties` | `List[str]` | `['energy', 'forces', 'stress']` | 计算性质列表 |

#### 返回

```python
{
    'energy': float,              # 总能量 (eV)
    'energy_per_atom': float,     # 每原子能量 (eV/atom)
    'forces': np.ndarray,         # 原子力 (N_atoms, 3) eV/Å
    'stress': np.ndarray,         # 应力张量 (6,) Voigt eV/Å³
    'pressure': float,            # 压力 (GPa)
    'max_force': float,           # 最大力 (eV/Å)
    'rms_force': float,           # RMS 力 (eV/Å)
    'virial': np.ndarray          # 维里张量 (3, 3) eV (可选)
}
```

#### 示例

```python
from ase.io import read

atoms = read('MOF-5.cif')
result = esen.single_point(atoms)

print(f"Energy: {result['energy']:.6f} eV")
print(f"Energy/atom: {result['energy_per_atom']:.6f} eV/atom")
print(f"Max force: {result['max_force']:.6f} eV/Å")
print(f"Pressure: {result['pressure']:.4f} GPa")
```

#### 性能

- **能量 MAE**: 0.041 eV/atom (**#1** 🥇)
- **力 MAE**: 0.084 eV/Å (#2)
- **应力 MAE**: 0.31 GPa (#3)

---

## 结构优化

### `optimize()`

通过最小化能量优化原子结构。

```python
def optimize(
    atoms: Atoms,
    fmax: float = 0.01,
    optimizer: str = 'LBFGS',
    relax_cell: bool = False,
    max_steps: int = 500,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    pressure: float = 0.0,
    hydrostatic_strain: bool = False
) -> Dict[str, Any]
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `atoms` | `ase.Atoms` | - | 待优化结构 |
| `fmax` | `float` | `0.01` | 收敛标准: max(\|F\|) < fmax (eV/Å) |
| `optimizer` | `str` | `'LBFGS'` | 优化器: `'LBFGS'`, `'BFGS'`, `'FIRE'` |
| `relax_cell` | `bool` | `False` | 是否优化晶胞参数 |
| `max_steps` | `int` | `500` | 最大优化步数 |
| `trajectory` | `str` | `None` | 轨迹文件路径 (`.traj`) |
| `logfile` | `str` | `None` | 日志文件路径 |
| `pressure` | `float` | `0.0` | 外压 (GPa) (仅 `relax_cell=True`) |
| `hydrostatic_strain` | `bool` | `False` | 是否仅各向同性应变 |

#### 返回

```python
{
    'converged': bool,            # 是否收敛
    'steps': int,                 # 实际优化步数
    'initial_energy': float,      # 初始能量 (eV)
    'final_energy': float,        # 最终能量 (eV)
    'energy_change': float,       # 能量变化 (eV)
    'initial_fmax': float,        # 初始最大力 (eV/Å)
    'final_fmax': float,          # 最终最大力 (eV/Å)
    'atoms': ase.Atoms,           # 优化后的结构
    'trajectory': List[Atoms]     # 优化轨迹 (如果 trajectory 为空字符串)
}
```

#### 示例

```python
from ase.io import read, write

atoms = read('MOF-5_initial.cif')

# 仅优化原子坐标
result = esen.optimize(
    atoms,
    fmax=0.01,
    optimizer='LBFGS',
    relax_cell=False,
    max_steps=500,
    trajectory='opt_coords.traj'
)

if result['converged']:
    print(f"Optimization converged in {result['steps']} steps")
    print(f"Energy降低: {result['energy_change']:.6f} eV")
    write('MOF-5_opt.cif', result['atoms'])
else:
    print("Warning: Optimization did not converge!")

# 全优化 (坐标 + 晶胞)
result_full = esen.optimize(
    atoms,
    fmax=0.01,
    relax_cell=True,
    pressure=0.0,    # 0 GPa (1 atm)
    max_steps=500,
    trajectory='opt_full.traj'
)

print(f"Volume change: {(result_full['atoms'].get_volume() - atoms.get_volume())/atoms.get_volume()*100:.2f}%")
```

#### 优化器选择

| 优化器 | 适用场景 | 收敛速度 | 内存需求 |
|--------|----------|----------|----------|
| `'LBFGS'` | **一般优化 (推荐)** | 快 | 中等 |
| `'BFGS'` | 小体系 (< 100 atoms) | 快 | 高 |
| `'FIRE'` | 难收敛体系 | 中等 | 低 |

#### 性能

- **成功率**: 89% (**#1** 🥇, 与 orb-v3-omat 并列)
- **平均步数**: ~150 (#2)

---

## 分子动力学

### `run_md()`

运行分子动力学模拟 (NVE/NVT/NPT)。

```python
def run_md(
    atoms: Atoms,
    temperature: float = 300.0,
    pressure: Optional[float] = None,
    steps: int = 10000,
    timestep: float = 1.0,
    ensemble: str = 'nvt',
    friction: float = 0.01,
    taut: Optional[float] = None,
    taup: Optional[float] = None,
    compressibility: Optional[float] = None,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
    log_interval: int = 100
) -> Atoms
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `atoms` | `ase.Atoms` | - | 初始结构 |
| `temperature` | `float` | `300.0` | 温度 (K) |
| `pressure` | `float` | `None` | 压力 (GPa) (仅 NPT) |
| `steps` | `int` | `10000` | MD 步数 |
| `timestep` | `float` | `1.0` | 时间步长 (fs) |
| `ensemble` | `str` | `'nvt'` | 系综: `'nve'`, `'nvt'`, `'npt'` |
| `friction` | `float` | `0.01` | Langevin 摩擦系数 (ps⁻¹) (NVT) |
| `taut` | `float` | `None` | 温度弛豫时间 (fs) (NPT, 默认 100) |
| `taup` | `float` | `None` | 压力弛豫时间 (fs) (NPT, 默认 1000) |
| `compressibility` | `float` | `None` | 压缩系数 (GPa⁻¹) (NPT) |
| `trajectory` | `str` | `None` | 轨迹文件路径 |
| `logfile` | `str` | `None` | 日志文件路径 |
| `log_interval` | `int` | `100` | 日志输出间隔 (步) |

#### 返回

- **`ase.Atoms`**: MD 模拟后的最终结构

#### 示例

```python
# NVT MD (300 K, 50 ps)
final_atoms = esen.run_md(
    atoms,
    temperature=300.0,
    steps=50000,       # 50,000 steps × 1 fs = 50 ps
    timestep=1.0,
    ensemble='nvt',
    friction=0.01,
    trajectory='nvt_300K.traj',
    logfile='nvt_300K.log',
    log_interval=100
)

# NPT MD (300 K, 1 atm, 100 ps)
final_atoms = esen.run_md(
    atoms,
    temperature=300.0,
    pressure=0.0,      # 0 GPa = 1 atm
    steps=100000,
    timestep=1.0,
    ensemble='npt',
    taut=100.0,
    taup=1000.0,
    compressibility=4.57e-5,  # MOF 典型值 (GPa⁻¹)
    trajectory='npt_300K_1atm.traj',
    logfile='npt_300K_1atm.log'
)

print(f"Final T: {final_atoms.get_temperature():.2f} K")
print(f"Final V: {final_atoms.get_volume():.2f} Å³")
```

#### 系综选择

| 系综 | 守恒量 | 适用场景 |
|------|--------|----------|
| `'nve'` | E (能量) | 微正则系综，测试能量守恒 |
| `'nvt'` | T (温度) | 恒温模拟 (Langevin) |
| `'npt'` | T, P (温度, 压力) | 恒温恒压，体系平衡 |

#### 性能

- **MD 稳定性**: **优异** (**#1** 🥇, 与 MatterSim 并列)
- **能量守恒**: 极佳 (#1)
- **长时间稳定**: 无结构坍塌

---

## 声子计算

### `phonon()`

使用 Phonopy 计算声子谱和热力学性质。

```python
def phonon(
    atoms: Atoms,
    supercell_matrix: Union[List[int], np.ndarray] = [2, 2, 2],
    mesh: Union[List[int], np.ndarray] = [20, 20, 20],
    displacement: float = 0.01,
    t_min: float = 0.0,
    t_max: float = 1000.0,
    t_step: float = 10.0
) -> Dict[str, Any]
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `atoms` | `ase.Atoms` | - | 原胞结构 (应充分优化) |
| `supercell_matrix` | `List[int]` | `[2, 2, 2]` | 超胞矩阵 (3×3 或 3 个整数) |
| `mesh` | `List[int]` | `[20, 20, 20]` | k 点网格 |
| `displacement` | `float` | `0.01` | 原子位移幅度 (Å) |
| `t_min` | `float` | `0.0` | 最低温度 (K) |
| `t_max` | `float` | `1000.0` | 最高温度 (K) |
| `t_step` | `float` | `10.0` | 温度步长 (K) |

#### 返回

```python
{
    'phonon': phonopy.Phonopy,    # Phonopy 对象
    'force_constants': np.ndarray,  # 力常数
    'frequency_points': np.ndarray,  # 频率点 (THz)
    'total_dos': np.ndarray,      # 总态密度
    'thermal': {
        'temperatures': np.ndarray,  # 温度 (K)
        'free_energy': np.ndarray,   # 自由能 (kJ/mol)
        'entropy': np.ndarray,       # 熵 (J/(K·mol))
        'heat_capacity': np.ndarray  # 热容 (J/(K·mol))
    },
    'has_imaginary': bool,        # 是否有虚频
    'imaginary_modes': int        # 虚频模式数量
}
```

#### 示例

```python
from ase.io import read
from esen_inference.tasks.phonon import plot_phonon_dos, plot_thermal_properties

# 加载充分优化的原胞
primitive = read('MOF-5_primitive_opt.cif')

# 声子计算 (2×2×2 超胞, 20×20×20 k-mesh)
result = esen.phonon(
    primitive,
    supercell_matrix=[2, 2, 2],
    mesh=[20, 20, 20],
    displacement=0.01,
    t_min=0,
    t_max=1000,
    t_step=10
)

# 检查虚频
if result['has_imaginary']:
    print(f"警告: 检测到 {result['imaginary_modes']} 个虚频!")
else:
    print("✓ 结构动力学稳定 (无虚频)")

# 绘制声子 DOS
plot_phonon_dos(
    result['frequency_points'],
    result['total_dos'],
    output='phonon_dos.png'
)

# 查看 300 K 热容
thermal = result['thermal']
idx_300K = (thermal['temperatures'] >= 300).argmax()
Cv_300K = thermal['heat_capacity'][idx_300K]
print(f"Cv at 300 K: {Cv_300K:.2f} J/(K·mol)")

# 绘制热容曲线
plot_thermal_properties(
    thermal['temperatures'],
    thermal['heat_capacity'],
    output='heat_capacity.png',
    mass_per_formula=1000.0  # MOF 摩尔质量 (g/mol)
)
```

#### 性能

- **热容 MAE**: 0.024 J/(K·g) (**#3** 🥉)
- **热容 MAPE**: 2.9% (#3)

---

## 力学性质

### `bulk_modulus()`

计算体积模量 (Bulk Modulus)。

```python
def bulk_modulus(
    atoms: Atoms,
    strain_range: float = 0.05,
    n_points: int = 7,
    eos_type: str = 'birchmurnaghan',
    optimize_first: bool = True,
    fmax: float = 0.01
) -> Dict[str, Any]
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `atoms` | `ase.Atoms` | - | 初始结构 |
| `strain_range` | `float` | `0.05` | 体积应变范围 (±5%) |
| `n_points` | `int` | `7` | 体积点数 (奇数, 包含 V₀) |
| `eos_type` | `str` | `'birchmurnaghan'` | EOS 类型: `'birchmurnaghan'`, `'murnaghan'`, `'vinet'` |
| `optimize_first` | `bool` | `True` | 是否先优化找 V₀ |
| `fmax` | `float` | `0.01` | 优化收敛标准 (eV/Å) |

#### 返回

```python
{
    'bulk_modulus': float,         # 体积模量 (GPa)
    'bulk_modulus_prime': float,   # B' (无量纲)
    'equilibrium_volume': float,   # V₀ (Å³)
    'equilibrium_energy': float,   # E₀ (eV)
    'eos': ase.eos.EquationOfState,  # EOS 对象
    'volumes': np.ndarray,         # 体积点 (Å³)
    'energies': np.ndarray         # 对应能量 (eV)
}
```

#### 示例

```python
from esen_inference.tasks.mechanics import plot_eos

atoms = read('MOF-5_opt.cif')

# 计算体积模量
result = esen.bulk_modulus(
    atoms,
    strain_range=0.05,    # ±5% 体积应变
    n_points=7,
    eos_type='birchmurnaghan',
    optimize_first=True,
    fmax=0.01
)

B = result['bulk_modulus']
V0 = result['equilibrium_volume']

print(f"Bulk modulus: {B:.2f} GPa")
print(f"Equilibrium volume: {V0:.3f} Å³")

# 绘制 EOS 曲线
plot_eos(
    result['volumes'],
    result['energies'],
    result['eos'],
    output='eos_birch_murnaghan.png'
)
```

#### 性能

- **体积模量 MAE**: **2.64 GPa** (**#1** 🥇)
- **EOS 拟合 R²**: 0.98+ (#1)

---

## 吸附能计算

### `adsorption_energy()`

计算客体分子在 MOF 中的吸附能。

```python
def adsorption_energy(
    host: Atoms,
    guest: Atoms,
    complex_atoms: Atoms,
    optimize_complex: bool = True,
    optimize_host: bool = False,
    optimize_guest: bool = False,
    fmax: float = 0.05
) -> Dict[str, Any]
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | `ase.Atoms` | - | 主体结构 (MOF) |
| `guest` | `ase.Atoms` | - | 客体分子 (CO₂, H₂O 等) |
| `complex_atoms` | `ase.Atoms` | - | 主-客复合物 |
| `optimize_complex` | `bool` | `True` | 是否优化复合物 |
| `optimize_host` | `bool` | `False` | 是否优化主体 |
| `optimize_guest` | `bool` | `False` | 是否优化客体 |
| `fmax` | `float` | `0.05` | 优化收敛标准 (eV/Å) |

#### 返回

```python
{
    'E_ads': float,               # 吸附能 (eV, 负值=稳定)
    'E_ads_per_atom': float,      # 每原子吸附能 (eV/atom)
    'E_complex': float,           # 复合物能量 (eV)
    'E_host': float,              # 主体能量 (eV)
    'E_guest': float,             # 客体能量 (eV)
    'optimized_complex': Atoms    # 优化后的复合物
}
```

#### 吸附能定义

```
E_ads = E_complex - (E_host + E_guest)
```

- **E_ads < 0**: 稳定吸附 (放热)
- **E_ads > 0**: 不稳定吸附 (吸热)

#### 示例

```python
from ase.io import read

# 加载结构
host = read('HKUST-1.cif')
guest = read('CO2.xyz')
complex_atoms = read('HKUST-1_CO2.cif')

# 计算吸附能
result = esen.adsorption_energy(
    host=host,
    guest=guest,
    complex_atoms=complex_atoms,
    optimize_complex=True,
    fmax=0.05
)

E_ads_eV = result['E_ads']
E_ads_kJ_mol = E_ads_eV * 96.485  # 转换为 kJ/mol

print(f"Adsorption energy: {E_ads_eV:.6f} eV")
print(f"Adsorption energy: {E_ads_kJ_mol:.2f} kJ/mol")

if E_ads_eV < 0:
    print("→ Stable adsorption (exothermic)")
else:
    print("→ Unstable adsorption (endothermic)")

# MOF 吸附能参考范围:
# CO₂: -10 to -40 kJ/mol (physisorption)
# H₂O: -40 to -80 kJ/mol (stronger interaction)
# H₂: -5 to -15 kJ/mol (weak interaction)
```

#### 性能

- **CO₂ 吸附**: 优异 (**#2** 🥈, 仅次于 MatterSim)
- **主客体相互作用**: 准确 (#2)

---

## 配位分析

### `coordination()`

分析金属中心的配位环境。

```python
def coordination(
    atoms: Atoms,
    center_indices: Optional[List[int]] = None,
    cutoff_scale: float = 1.3,
    neighbor_indices: Optional[List[int]] = None
) -> Dict[str, Any]
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `atoms` | `ase.Atoms` | - | 原子结构 |
| `center_indices` | `List[int]` | `None` | 中心原子索引 (None=所有原子) |
| `cutoff_scale` | `float` | `1.3` | 截断缩放因子 × 共价半径和 |
| `neighbor_indices` | `List[int]` | `None` | 配体原子索引 (None=所有原子) |

#### 返回

```python
{
    'coordination_numbers': Dict[int, int],      # {原子索引: 配位数}
    'neighbor_lists': Dict[int, List[int]],      # {原子索引: [邻居索引]}
    'distances': Dict[int, List[float]],         # {原子索引: [距离 (Å)]}
    'neighbor_symbols': Dict[int, List[str]]     # {原子索引: [邻居元素]}
}
```

#### 示例

```python
from ase.io import read
from collections import Counter

atoms = read('HKUST-1.cif')

# 找到所有 Cu 原子
cu_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == 'Cu']

# 配位分析
result = esen.coordination(
    atoms,
    center_indices=cu_indices,
    cutoff_scale=1.3
)

cn = result['coordination_numbers']
neighbor_lists = result['neighbor_lists']
distances = result['distances']
neighbor_symbols = result['neighbor_symbols']

# 显示前 5 个 Cu 的配位环境
for cu_idx in cu_indices[:5]:
    print(f"\nCu atom {cu_idx}:")
    print(f"  Coordination number: {cn[cu_idx]}")
    print(f"  Neighbors: {neighbor_lists[cu_idx]}")
    print(f"  Distances (Å): {[f'{d:.3f}' for d in distances[cu_idx]]}")
    
    # 配位原子类型统计
    coord_types = Counter(neighbor_symbols[cu_idx])
    print(f"  Coordination types: {dict(coord_types)}")
    # 例如: {'O': 4, 'C': 1} → 四配位 O + 一配位 C
```

---

## 工具函数

### I/O 工具

#### `read_structure()`

```python
from esen_inference.utils.io import read_structure

atoms = read_structure('MOF-5.cif')  # 自动识别格式
atoms = read_structure('trajectory.xyz', index=':')  # 读取全部轨迹
atoms = read_structure('POSCAR')  # VASP 格式
```

#### `write_structure()`

```python
from esen_inference.utils.io import write_structure

write_structure(atoms, 'output.cif')
write_structure(atoms, 'output.xyz')
write_structure(atoms, 'POSCAR', format='vasp')
```

### 可视化工具

#### `plot_phonon_dos()`

```python
from esen_inference.tasks.phonon import plot_phonon_dos

plot_phonon_dos(
    frequency_points,  # THz
    total_dos,
    output='phonon_dos.png',
    title='Phonon DOS',
    xlim=(0, 50),      # THz
    figsize=(8, 6)
)
```

#### `plot_thermal_properties()`

```python
from esen_inference.tasks.phonon import plot_thermal_properties

plot_thermal_properties(
    temperatures,      # K
    heat_capacity,     # J/(K·mol)
    output='Cv.png',
    title='Heat Capacity',
    mass_per_formula=1000.0  # g/mol
)
```

#### `plot_eos()`

```python
from esen_inference.tasks.mechanics import plot_eos

plot_eos(
    volumes,           # Å³
    energies,          # eV
    eos_object,        # ASE EOS
    output='eos.png',
    title='Equation of State'
)
```

### MD 分析工具

#### `analyze_md_trajectory()`

```python
from esen_inference.tasks.dynamics import analyze_md_trajectory
from ase.io import read

trajectory = read('md.traj', ':')

analysis = analyze_md_trajectory(trajectory)

print(f"平均温度: {analysis['avg_temperature']:.2f} K")
print(f"温度标准差: {analysis['std_temperature']:.2f} K")
print(f"平均体积: {analysis['avg_volume']:.2f} Å³")
print(f"能量漂移: {analysis['energy_drift']:.6f} eV")
print(f"MSD (最终): {analysis['msd'][-1]:.4f} Å²")
```

返回:
```python
{
    'avg_temperature': float,
    'std_temperature': float,
    'avg_volume': float,
    'std_volume': float,
    'avg_energy': float,
    'energy_drift': float,
    'msd': np.ndarray  # Mean squared displacement
}
```

---

## 设备管理

### `set_device()`

动态切换计算设备。

```python
# 初始化时使用 GPU
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# 切换到 CPU (例如 GPU 内存不足时)
esen.set_device('cpu')

# 切换到特定 GPU
esen.set_device('cuda:1')
```

### `get_device_info()`

查看当前设备信息。

```python
from esen_inference.utils.device import get_device_info

info = get_device_info()

print(f"Device type: {info['device_type']}")
print(f"Device name: {info['device_name']}")

if info['device_type'] == 'cuda':
    print(f"CUDA version: {info['cuda_version']}")
    print(f"GPU memory: {info['gpu_memory_total']} MB")
    print(f"GPU memory free: {info['gpu_memory_free']} MB")
```

### 批处理优化

```python
import torch

# 清理 GPU 缓存 (批处理前)
torch.cuda.empty_cache()

# 对于大体系，使用 CPU
esen_cpu = ESENInference(model_name='esen-30m-oam', device='cpu', cpu_threads=16)

# 或降低精度
esen_fp32 = ESENInference(model_name='esen-30m-oam', device='cuda', precision='float32')
```

---

## 命令行接口 (CLI)

### `esen-infer`

```bash
# 单点能量计算
esen-infer single-point MOF-5.cif --output result.json

# 结构优化
esen-infer optimize MOF-5.cif --fmax 0.01 --relax-cell --output MOF-5_opt.cif

# 批量优化
esen-infer batch-optimize mof_database/*.cif --output-dir optimized/

# 声子计算
esen-infer phonon MOF-5_primitive.cif --supercell 2 2 2 --mesh 20 20 20

# 体积模量
esen-infer bulk-modulus MOF-5_opt.cif --strain-range 0.05 --n-points 7

# MD 模拟
esen-infer md MOF-5_opt.cif --temperature 300 --steps 50000 --ensemble nvt
```

详细 CLI 使用请参考 `esen-infer --help`。

---

## 性能总结

### MOFSimBench 排名

| 任务 | eSEN-OAM 排名 | MAE/指标 |
|------|--------------|----------|
| **能量预测** | **#1** 🥇 | 0.041 eV/atom |
| **体积模量** | **#1** 🥇 | 2.64 GPa |
| **结构优化** | **#1** 🥇 | 89% 成功率 |
| **MD 稳定性** | **#1** 🥇 | 优异 |
| **力预测** | #2 🥈 | 0.084 eV/Å |
| **吸附能** | #2 🥈 | 优异 |
| **热容** | #3 🥉 | 0.024 J/(K·g) |
| **应力预测** | #3 🥉 | 0.31 GPa |

**结论**: eSEN-30M-OAM 是 MOFSimBench **整体排名第一** 的模型，在 **能量、力学、优化、MD** 任务中表现最佳。

---

## 最佳实践

### 1. 推荐工作流

```python
from esen_inference import ESENInference
from ase.io import read, write

# 1. 初始化 (一次性)
esen = ESENInference(model_name='esen-30m-oam', device='cuda')

# 2. 结构优化
atoms = read('MOF-5_initial.cif')
opt_result = esen.optimize(atoms, fmax=0.01, relax_cell=True)
write('MOF-5_opt.cif', opt_result['atoms'])

# 3. 单点性质
sp_result = esen.single_point(opt_result['atoms'])

# 4. 力学性质
bulk_result = esen.bulk_modulus(opt_result['atoms'], optimize_first=False)

# 5. 声子 & 热容
phonon_result = esen.phonon(opt_result['atoms'], supercell_matrix=[2, 2, 2])

# 6. MD 模拟
md_final = esen.run_md(opt_result['atoms'], temperature=300, steps=50000, ensemble='nvt')
```

### 2. GPU 内存优化

```python
import torch

# 大体系策略 1: 降低精度
esen = ESENInference(model_name='esen-30m-oam', device='cuda', precision='float32')

# 大体系策略 2: 使用 CPU
esen_cpu = ESENInference(model_name='esen-30m-oam', device='cpu', cpu_threads=16)

# 批处理: 定期清理缓存
for mof_file in mof_files:
    result = esen.optimize(read(mof_file))
    torch.cuda.empty_cache()  # 每个 MOF 后清理
```

### 3. 高精度计算

```python
# 两阶段优化
# Stage 1: 粗优化
result1 = esen.optimize(atoms, fmax=0.05, relax_cell=True)

# Stage 2: 精优化 (float64)
esen_hp = ESENInference(model_name='esen-30m-oam', device='cuda', precision='float64')
result2 = esen_hp.optimize(result1['atoms'], fmax=0.001, relax_cell=True)
```

---

**文档版本**: v1.0  
**更新日期**: 2026-01-07  
**API 版本**: esen_inference v1.0.0  
**模型版本**: eSEN-30M-OAM / eSEN-30M-MP  
**核心依赖**: fairchem (FAIR-Chem), ASE, Phonopy
