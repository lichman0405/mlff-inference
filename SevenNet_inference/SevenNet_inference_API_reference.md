# SevenNet Inference - API Reference

> **SevenNet**: MOFSimBench 排名 **#4** 的通用机器学习力场  
> **特色**: 等变GNN、力预测精度高、计算高效、7层网络架构

---

## 目录

1. [核心类: SevenNetInference](#1-核心类-sevennetinference)
2. [Utils 模块](#2-utils-模块)
3. [Tasks 模块](#3-tasks-模块)
4. [CLI 命令行工具](#4-cli-命令行工具)
5. [异常处理](#5-异常处理)

---

## 1. 核心类: SevenNetInference

### 1.1 类定义

```python
class SevenNetInference:
    """
    SevenNet 推理主类。
    
    封装 SevenNet 模型,提供材料性质计算的统一接口。
    SevenNet 是基于等变图神经网络(Equivariant GNN)的通用力场,
    在力预测方面表现优异,计算效率高。
    
    Attributes:
        model_name: 模型名称
        device: 计算设备 ('cuda' / 'cpu')
        calculator: ASE Calculator 实例
        model_path: 模型权重路径
    
    Example:
        >>> from sevennet_inference import SevenNetInference
        >>> calc = SevenNetInference(model_name="SevenNet-0", device="cuda")
        >>> print(calc)
        SevenNetInference(model=SevenNet-0, device=cuda)
    """
```

### 1.2 初始化

```python
def __init__(
    self,
    model_name: str = "SevenNet-0",
    device: str = "auto",
    model_path: Optional[str] = None,
    cutoff: float = 6.0,
    max_neighbors: int = 100,
    **kwargs
) -> None:
    """
    初始化 SevenNetInference。
    
    Args:
        model_name: 模型名称
            - "SevenNet-0": 标准版 (~2M 参数) [默认,推荐]
            - "SevenNet-0-22May2024": 最新检查点
        device: 计算设备
            - "auto": 自动检测 (优先 GPU) [默认]
            - "cuda": 强制使用 GPU
            - "cpu": 强制使用 CPU
        model_path: 自定义模型路径 (可选)
            - 如果提供,将覆盖 model_name
        cutoff: 截断半径 (Å)
            - 默认: 6.0 Å
            - 影响近邻搜索范围
        max_neighbors: 最大近邻数
            - 默认: 100
            - 影响内存使用
        **kwargs: 传递给底层 Calculator 的额外参数
            - precision: 'float32' / 'mixed' / 'float16'
            - enable_backward: bool (梯度计算)
    
    Returns:
        None
    
    Raises:
        ModelNotFoundError: 模型未找到
        DeviceError: 设备不可用
    
    Example:
        >>> # 基本用法
        >>> calc = SevenNetInference()
        
        >>> # 指定设备
        >>> calc = SevenNetInference(model_name="SevenNet-0", device="cuda")
        
        >>> # 使用自定义模型
        >>> calc = SevenNetInference(
        ...     model_path="/path/to/custom_model.pth",
        ...     device="cuda"
        ... )
        
        >>> # 调整截断半径和精度
        >>> calc = SevenNetInference(
        ...     model_name="SevenNet-0",
        ...     device="cuda",
        ...     cutoff=7.0,
        ...     precision="mixed"
        ... )
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
    单点能量和力计算。
    
    SevenNet 在力预测方面表现优异 (MAE = 0.102 eV/Å, #4 in MOFSimBench)。
    
    Args:
        atoms: 输入结构
            - ASE Atoms 对象
            - 结构文件路径 (CIF, POSCAR, XYZ 等)
    
    Returns:
        dict: 包含以下键:
            - energy: 总能量 (eV)
            - energy_per_atom: 每原子能量 (eV/atom)
            - forces: 原子力 (N, 3) (eV/Å)
            - stress: 应力张量 (6,) (eV/Å³)
                格式: [xx, yy, zz, yz, xz, xy]
            - max_force: 最大力分量 (eV/Å)
            - rms_force: RMS 力 (eV/Å)
            - pressure: 压强 (GPa)
                计算: -trace(stress)/3
    
    Raises:
        StructureError: 结构读取或验证失败
        CalculationError: 计算失败
    
    Example:
        >>> from ase.io import read
        >>> atoms = read("MOF-5.cif")
        >>> result = calc.single_point(atoms)
        >>> 
        >>> print(f"能量: {result['energy']:.4f} eV")
        >>> print(f"最大力: {result['max_force']:.4f} eV/Å")
        >>> 
        >>> # 访问详细数据
        >>> forces = result['forces']
        >>> print(f"原子0的力: {forces[0]}")
        >>> 
        >>> # 直接传入文件路径
        >>> result = calc.single_point("structure.cif")
    
    Notes:
        - 力的精度: SevenNet 力 MAE = 0.102 eV/Å (MOFSimBench #4)
        - 应力张量采用 Voigt 记号
        - 压强定义: P = -Tr(σ)/3
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
    trajectory: Optional[str] = None,
    output: Optional[str] = None,
    logfile: Optional[str] = None
) -> Dict[str, Any]:
    """
    结构优化。
    
    利用 SevenNet 的高力预测精度实现高效结构优化。
    
    Args:
        atoms: 输入结构
            - ASE Atoms 对象
            - 结构文件路径
        fmax: 力收敛阈值 (eV/Å)
            - 默认: 0.05 eV/Å (标准精度)
            - 推荐范围: 0.01 - 0.10
        optimizer: 优化器类型
            - "LBFGS": 拟牛顿法 [默认,推荐]
            - "BFGS": BFGS 算法
            - "FIRE": 快速惯性松弛引擎
        optimize_cell: 是否同时优化晶胞
            - False: 仅优化原子坐标 [默认]
            - True: 同时优化晶胞和坐标
        max_steps: 最大优化步数
            - 默认: 500
        trajectory: 轨迹文件路径 (可选)
            - 保存每一步的结构
            - 格式: .traj (ASE trajectory)
        output: 输出文件路径 (可选)
            - 保存最终优化结构
            - 支持: CIF, POSCAR, XYZ 等
        logfile: 日志文件路径 (可选)
            - 记录优化过程
    
    Returns:
        dict: 包含以下键:
            - converged: 是否收敛 (bool)
            - steps: 实际优化步数 (int)
            - initial_energy: 初始能量 (eV)
            - final_energy: 最终能量 (eV)
            - energy_change: 能量变化 (eV)
            - initial_fmax: 初始最大力 (eV/Å)
            - final_fmax: 最终最大力 (eV/Å)
            - atoms: 优化后的 Atoms 对象
            - initial_volume: 初始体积 (Å³) [仅 optimize_cell=True]
            - final_volume: 最终体积 (Å³) [仅 optimize_cell=True]
            - volume_change: 体积变化百分比 (%) [仅 optimize_cell=True]
    
    Raises:
        ConvergenceError: 未在最大步数内收敛
        StructureError: 结构问题
    
    Example:
        >>> # 仅优化原子坐标
        >>> result = calc.optimize(atoms, fmax=0.05)
        >>> print(f"收敛: {result['converged']}")
        >>> print(f"步数: {result['steps']}")
        >>> 
        >>> # 同时优化晶胞
        >>> result = calc.optimize(
        ...     atoms,
        ...     fmax=0.01,
        ...     optimize_cell=True,
        ...     output='optimized.cif'
        ... )
        >>> 
        >>> # 保存优化轨迹
        >>> result = calc.optimize(
        ...     atoms,
        ...     fmax=0.05,
        ...     trajectory='opt.traj',
        ...     logfile='opt.log'
        ... )
        >>> 
        >>> # 获取优化后的结构
        >>> optimized_atoms = result['atoms']
        >>> optimized_atoms.write('final.cif')
    
    Notes:
        - LBFGS 通常最快,推荐作为默认
        - FIRE 对复杂势能面更稳定
        - fmax=0.01 适合高精度计算
        - optimize_cell=True 时计算时间增加约2-3倍
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
    log_interval: int = 100,
    friction: float = 0.002
) -> Atoms:
    """
    分子动力学模拟。
    
    SevenNet 具有良好的 MD 稳定性,适合中等时长的 MD 模拟。
    
    Args:
        atoms: 输入结构
            - ASE Atoms 对象
            - 结构文件路径
        ensemble: 系综类型
            - "nve": 微正则系综 (恒 E, V)
            - "nvt": 正则系综 (恒 T, V) [默认]
            - "npt": 等温等压系综 (恒 T, P)
        temperature: 温度 (K)
            - 默认: 300.0 K
            - NVE 系综会忽略此参数
        pressure: 压强 (GPa)
            - 仅 NPT 系综需要
            - 0.0 = 1 atm
            - None = 未指定
        steps: 模拟步数
            - 默认: 10000
            - 总时间 = steps × timestep
        timestep: 时间步长 (fs)
            - 默认: 1.0 fs
            - 含氢体系推荐: 0.5 fs
            - 高温 (>500K) 推荐: 0.5 fs
        trajectory: 轨迹文件路径 (可选)
            - 格式: .traj (ASE trajectory)
        logfile: 日志文件路径 (可选)
            - 记录能量、温度等
        log_interval: 日志记录间隔 (步)
            - 默认: 100
            - 每 log_interval 步记录一次
        friction: Langevin 摩擦系数 (1/fs)
            - 默认: 0.002
            - 仅 NVT 系综使用
            - 控制热浴耦合强度
    
    Returns:
        Atoms: 模拟结束时的结构
    
    Raises:
        MDError: MD 模拟失败或不稳定
    
    Example:
        >>> # NVT 模拟
        >>> final = calc.run_md(
        ...     atoms,
        ...     ensemble='nvt',
        ...     temperature=300,
        ...     steps=50000,
        ...     trajectory='nvt.traj',
        ...     logfile='nvt.log'
        ... )
        >>> 
        >>> # NPT 模拟
        >>> final = calc.run_md(
        ...     atoms,
        ...     ensemble='npt',
        ...     temperature=300,
        ...     pressure=0.0,  # 1 atm
        ...     steps=100000,
        ...     timestep=1.0
        ... )
        >>> 
        >>> # 高温 NVT
        >>> final = calc.run_md(
        ...     atoms,
        ...     ensemble='nvt',
        ...     temperature=500,
        ...     timestep=0.5,  # 更小的步长
        ...     steps=50000
        ... )
        >>> 
        >>> # 分析轨迹
        >>> from ase.io import read
        >>> traj = read('nvt.traj', index=':')
        >>> temps = [a.get_temperature() for a in traj]
    
    Notes:
        - NVT 使用 Langevin 热浴
        - NPT 使用 Berendsen 气压浴
        - 推荐先用 NVT 平衡再用 NPT
        - 高温或含氢体系需要更小的时间步长
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
    t_step: float = 10,
    imaginary_threshold: float = -0.1
) -> Dict[str, Any]:
    """
    声子态密度和热力学性质计算。
    
    使用有限位移法(Finite Displacement Method)计算声子性质。
    
    Args:
        atoms: 原胞结构
            - 必须是原胞,不能是超胞
            - ASE Atoms 对象或文件路径
        supercell_matrix: 超胞扩展矩阵
            - 格式: [a, b, c]
            - 默认: [2, 2, 2]
            - 影响精度和计算量
        mesh: k 点网格
            - 格式: [kx, ky, kz]
            - 默认: [20, 20, 20]
            - 影响态密度精度
        displacement: 原子位移大小 (Å)
            - 默认: 0.01 Å
            - 用于数值微分
        t_min: 最低温度 (K)
            - 默认: 0
        t_max: 最高温度 (K)
            - 默认: 1000
        t_step: 温度步长 (K)
            - 默认: 10
        imaginary_threshold: 虚频判断阈值 (THz)
            - 默认: -0.1 THz
            - 频率 < threshold 视为虚频
    
    Returns:
        dict: 包含以下键:
            - frequency_points: 频率点数组 (THz)
            - total_dos: 总态密度
            - has_imaginary: 是否存在虚频 (bool)
            - imaginary_modes: 虚频数量 (int)
            - thermal: 热力学性质字典
                - temperatures: 温度数组 (K)
                - heat_capacity: 等容热容 (J/(mol·K))
                - entropy: 熵 (J/(mol·K))
                - free_energy: 自由能 (kJ/mol)
            - supercell_size: 实际使用的超胞大小
    
    Raises:
        PhononError: 声子计算失败
        StructureError: 结构不是原胞
    
    Example:
        >>> # 基本声子计算
        >>> result = calc.phonon(
        ...     primitive,
        ...     supercell_matrix=[2, 2, 2],
        ...     mesh=[20, 20, 20]
        ... )
        >>> 
        >>> # 检查虚频
        >>> if result['has_imaginary']:
        ...     print(f"警告: {result['imaginary_modes']} 个虚频")
        ... else:
        ...     print("✓ 结构稳定")
        >>> 
        >>> # 获取 300K 热容
        >>> thermal = result['thermal']
        >>> idx_300 = list(thermal['temperatures']).index(300)
        >>> cv_300 = thermal['heat_capacity'][idx_300]
        >>> print(f"300K 热容: {cv_300:.3f} J/(mol·K)")
        >>> 
        >>> # 绘制声子态密度
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(result['frequency_points'], result['total_dos'])
        >>> plt.xlabel('Frequency (THz)')
        >>> plt.ylabel('DOS')
        >>> plt.savefig('phonon_dos.png')
    
    Notes:
        - 计算量 ∝ 超胞原子数 × 位移数
        - 大型 MOF 建议使用较小的超胞
        - 虚频可能表示结构不稳定
        - 需要先优化结构再计算声子
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
    
    通过拟合能量-体积(E-V)曲线计算体模量。
    
    Args:
        atoms: 输入结构
            - ASE Atoms 对象
            - 结构文件路径
        strain_range: 应变范围 (±)
            - 默认: 0.05 (±5%)
            - 体积变化范围: (1-ε)³V₀ 到 (1+ε)³V₀
        npoints: 采样点数
            - 默认: 11
            - 更多点提高拟合精度
        eos: 状态方程类型
            - "birchmurnaghan": Birch-Murnaghan [默认]
            - "vinet": Vinet EOS
            - "murnaghan": Murnaghan EOS
    
    Returns:
        dict: 包含以下键:
            - bulk_modulus: 体模量 (GPa)
            - v0: 平衡体积 (Å³)
            - e0: 平衡能量 (eV)
            - b0_prime: 体模量导数
            - eos: 使用的 EOS 类型
            - volumes: 体积数组 (Å³)
            - energies: 能量数组 (eV)
            - fit_quality: 拟合质量 (R²)
    
    Raises:
        FitError: EOS 拟合失败
    
    Example:
        >>> # 基本体模量计算
        >>> result = calc.bulk_modulus(atoms)
        >>> print(f"体模量: {result['bulk_modulus']:.2f} GPa")
        >>> 
        >>> # 使用不同的 EOS
        >>> result = calc.bulk_modulus(
        ...     atoms,
        ...     strain_range=0.08,
        ...     eos='vinet'
        ... )
        >>> 
        >>> # 绘制 E-V 曲线
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(result['volumes'], result['energies'], 'o-')
        >>> plt.xlabel('Volume (Å³)')
        >>> plt.ylabel('Energy (eV)')
        >>> plt.title(f"B = {result['bulk_modulus']:.2f} GPa")
        >>> plt.savefig('eos_curve.png')
    
    Notes:
        - 建议先优化结构
        - strain_range 不宜过大 (< 0.1)
        - npoints ≥ 7 保证拟合质量
    """
```

#### 1.3.6 elastic_constants

```python
def elastic_constants(
    self,
    atoms: Union[Atoms, str, Path],
    symmetry: str = "cubic",
    delta: float = 0.01
) -> Dict[str, Any]:
    """
    弹性常数计算。
    
    通过应变-应力关系计算弹性张量。
    
    Args:
        atoms: 输入结构
        symmetry: 晶体对称性
            - "cubic": 立方
            - "hexagonal": 六方
            - "orthorhombic": 正交
            - "triclinic": 三斜
        delta: 应变增量
            - 默认: 0.01 (1%)
    
    Returns:
        dict: 包含以下键:
            - elastic_tensor: 弹性张量 (6×6) (GPa)
            - bulk_modulus: 体模量 (GPa)
            - shear_modulus: 剪切模量 (GPa)
            - youngs_modulus: 杨氏模量 (GPa)
            - poisson_ratio: 泊松比
    
    Example:
        >>> result = calc.elastic_constants(atoms, symmetry="cubic")
        >>> C = result['elastic_tensor']
        >>> print(f"C11 = {C[0,0]:.2f} GPa")
        >>> print(f"体模量 = {result['bulk_modulus']:.2f} GPa")
    """
```

---

## 2. Utils 模块

### 2.1 device.py

#### 2.1.1 get_device

```python
def get_device(device: str = "auto") -> str:
    """
    获取计算设备。
    
    Args:
        device: 设备选择
            - "auto": 自动检测 (优先 GPU)
            - "cuda": 强制 GPU
            - "cuda:0", "cuda:1": 指定 GPU
            - "cpu": 强制 CPU
    
    Returns:
        str: 实际使用的设备名称
    
    Raises:
        DeviceError: 请求的设备不可用
    
    Example:
        >>> device = get_device("auto")
        >>> print(device)  # 'cuda:0' 或 'cpu'
        >>> 
        >>> # 强制使用特定 GPU
        >>> device = get_device("cuda:1")
    """
```

#### 2.1.2 get_available_devices

```python
def get_available_devices() -> List[str]:
    """
    获取所有可用设备列表。
    
    Returns:
        List[str]: 可用设备列表
    
    Example:
        >>> devices = get_available_devices()
        >>> print(devices)
        ['cpu', 'cuda:0', 'cuda:1']
    """
```

#### 2.1.3 print_device_info

```python
def print_device_info() -> None:
    """
    打印设备信息。
    
    显示 CPU 和 GPU 的详细信息。
    
    Example:
        >>> print_device_info()
        Device Information:
        ------------------
        CPU: Intel Core i9-12900K
        GPU 0: NVIDIA RTX 4090 (24GB)
        GPU 1: NVIDIA RTX 3090 (24GB)
    """
```

### 2.2 io.py

#### 2.2.1 read_structure

```python
def read_structure(filepath: Union[str, Path]) -> Atoms:
    """
    读取结构文件。
    
    支持格式: CIF, POSCAR, XYZ, EXTXYZ, PDB, LAMMPS-data
    
    Args:
        filepath: 文件路径
    
    Returns:
        Atoms: ASE Atoms 对象
    
    Raises:
        FileNotFoundError: 文件不存在
        FormatError: 格式不支持
    
    Example:
        >>> atoms = read_structure("MOF-5.cif")
        >>> atoms = read_structure("POSCAR")
    """
```

#### 2.2.2 write_structure

```python
def write_structure(
    atoms: Atoms,
    filepath: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    写入结构文件。
    
    Args:
        atoms: ASE Atoms 对象
        filepath: 输出路径
        format: 文件格式 (可选,自动从扩展名推断)
        **kwargs: 传递给 ase.io.write 的参数
    
    Example:
        >>> write_structure(atoms, "output.cif")
        >>> write_structure(atoms, "POSCAR", format="vasp")
    """
```

#### 2.2.3 validate_structure

```python
def validate_structure(atoms: Atoms) -> Tuple[bool, str]:
    """
    验证结构有效性。
    
    检查:
    - 是否有原子
    - 是否有晶胞(周期性体系)
    - 原子距离是否合理
    
    Args:
        atoms: ASE Atoms 对象
    
    Returns:
        Tuple[bool, str]: (是否有效, 错误信息)
    
    Example:
        >>> valid, msg = validate_structure(atoms)
        >>> if not valid:
        ...     print(f"结构无效: {msg}")
    """
```

### 2.3 analysis.py

#### 2.3.1 analyze_trajectory

```python
def analyze_trajectory(
    trajectory_file: str
) -> Dict[str, Any]:
    """
    分析 MD 轨迹。
    
    Args:
        trajectory_file: 轨迹文件路径 (.traj)
    
    Returns:
        dict: 分析结果
            - mean_temperature: 平均温度 (K)
            - std_temperature: 温度标准差 (K)
            - mean_energy: 平均能量 (eV)
            - std_energy: 能量标准差 (eV)
            - mean_volume: 平均体积 (Å³)
            - msd: 均方位移
    
    Example:
        >>> result = analyze_trajectory("md.traj")
        >>> print(f"平均温度: {result['mean_temperature']:.2f} K")
    """
```

---

## 3. Tasks 模块

### 3.1 static.py

```python
def calculate_single_point(
    atoms: Atoms,
    calculator: Any
) -> Dict[str, Any]:
    """
    执行单点计算。
    
    底层函数,被 SevenNetInference.single_point() 调用。
    
    Args:
        atoms: ASE Atoms 对象
        calculator: ASE Calculator
    
    Returns:
        dict: 计算结果
    """
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
    """
    运行分子动力学。
    
    底层函数,被 SevenNetInference.run_md() 调用。
    
    Args:
        atoms: ASE Atoms 对象
        calculator: ASE Calculator
        ensemble: 系综类型
        temperature: 温度 (K)
        **kwargs: 额外参数
    
    Returns:
        Atoms: 最终结构
    """
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
    """
    计算声子性质。
    
    使用 Phonopy 进行声子计算。
    
    Args:
        atoms: 原胞
        calculator: ASE Calculator
        supercell_matrix: 超胞大小
        mesh: k 点网格
        **kwargs: 额外参数
    
    Returns:
        dict: 声子结果
    """
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
    """
    计算体模量。
    
    通过 EOS 拟合。
    
    Args:
        atoms: ASE Atoms 对象
        calculator: ASE Calculator
        strain_range: 应变范围
        npoints: 采样点数
        eos: EOS 类型
    
    Returns:
        dict: 体模量结果
    """
```

---

## 4. CLI 命令行工具

### 4.1 命令概览

```bash
sevennet-infer <command> [options]
```

### 4.2 可用命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `single-point` | 单点计算 | `sevennet-infer single-point MOF.cif` |
| `optimize` | 结构优化 | `sevennet-infer optimize MOF.cif --fmax 0.01` |
| `md` | 分子动力学 | `sevennet-infer md MOF.cif --ensemble nvt` |
| `phonon` | 声子计算 | `sevennet-infer phonon prim.cif --supercell 2 2 2` |
| `bulk-modulus` | 体模量 | `sevennet-infer bulk-modulus MOF.cif` |
| `batch-optimize` | 批量优化 | `sevennet-infer batch-optimize *.cif` |

### 4.3 命令详解

#### 4.3.1 single-point

```bash
sevennet-infer single-point <structure> [options]

选项:
  --output, -o PATH       输出 JSON 文件
  --device DEVICE         设备 (auto/cuda/cpu)
  --model MODEL           模型名称
  --verbose, -v           详细输出

示例:
  sevennet-infer single-point MOF-5.cif
  sevennet-infer single-point MOF-5.cif --output result.json
  sevennet-infer single-point MOF-5.cif --device cuda --verbose
```

#### 4.3.2 optimize

```bash
sevennet-infer optimize <structure> [options]

选项:
  --fmax FLOAT            力收敛阈值 (默认: 0.05)
  --optimizer OPT         优化器 (LBFGS/BFGS/FIRE)
  --cell                  同时优化晶胞
  --max-steps INT         最大步数 (默认: 500)
  --output, -o PATH       输出结构文件
  --trajectory PATH       轨迹文件
  --device DEVICE         设备

示例:
  sevennet-infer optimize MOF.cif --fmax 0.05
  sevennet-infer optimize MOF.cif --fmax 0.01 --cell -o opt.cif
  sevennet-infer optimize MOF.cif --optimizer FIRE --trajectory opt.traj
```

#### 4.3.3 md

```bash
sevennet-infer md <structure> [options]

选项:
  --ensemble ENSEMBLE     系综 (nve/nvt/npt)
  --temp, -T FLOAT        温度 (K)
  --pressure, -P FLOAT    压强 (GPa, NPT)
  --steps INT             模拟步数
  --timestep FLOAT        时间步长 (fs)
  --trajectory PATH       轨迹文件
  --logfile PATH          日志文件
  --device DEVICE         设备

示例:
  sevennet-infer md MOF.cif --ensemble nvt --temp 300 --steps 50000
  sevennet-infer md MOF.cif --ensemble npt --temp 300 --pressure 0.0
  sevennet-infer md MOF.cif --ensemble nvt -T 500 --timestep 0.5
```

#### 4.3.4 phonon

```bash
sevennet-infer phonon <structure> [options]

选项:
  --supercell INT INT INT  超胞大小
  --mesh INT INT INT       k 点网格
  --t-min FLOAT            最低温度 (K)
  --t-max FLOAT            最高温度 (K)
  --t-step FLOAT           温度步长 (K)
  --output, -o PATH        输出文件
  --device DEVICE          设备

示例:
  sevennet-infer phonon prim.cif --supercell 2 2 2
  sevennet-infer phonon prim.cif --supercell 3 3 3 --mesh 30 30 30
  sevennet-infer phonon prim.cif --supercell 2 2 2 --t-max 1000
```

#### 4.3.5 bulk-modulus

```bash
sevennet-infer bulk-modulus <structure> [options]

选项:
  --strain-range FLOAT    应变范围 (默认: 0.05)
  --npoints INT           采样点数 (默认: 11)
  --eos EOS               EOS 类型
  --output, -o PATH       输出文件
  --device DEVICE         设备

示例:
  sevennet-infer bulk-modulus MOF.cif
  sevennet-infer bulk-modulus MOF.cif --strain-range 0.08
  sevennet-infer bulk-modulus MOF.cif --eos vinet --output bm.json
```

#### 4.3.6 batch-optimize

```bash
sevennet-infer batch-optimize <patterns> [options]

选项:
  --fmax FLOAT            力收敛阈值
  --cell                  优化晶胞
  --output-dir PATH       输出目录
  --report PATH           报告文件 (CSV)
  --device DEVICE         设备

示例:
  sevennet-infer batch-optimize *.cif --output-dir optimized/
  sevennet-infer batch-optimize mofs/*.cif --fmax 0.05 --cell
  sevennet-infer batch-optimize *.cif --report results.csv
```

### 4.4 全局选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称 | SevenNet-0 |
| `--device` | 设备 | auto |
| `--verbose, -v` | 详细输出 | False |
| `--help, -h` | 帮助信息 | - |
| `--version` | 版本信息 | - |

---

## 5. 异常处理

### 5.1 异常类

```python
from sevennet_inference.exceptions import (
    SevenNetError,          # 基类
    ModelNotFoundError,     # 模型未找到
    DeviceError,            # 设备错误
    StructureError,         # 结构错误
    CalculationError,       # 计算错误
    ConvergenceError,       # 收敛错误
    PhononError,            # 声子计算错误
    MDError                 # MD 错误
)
```

### 5.2 使用示例

```python
from sevennet_inference import SevenNetInference
from sevennet_inference.exceptions import *

try:
    calc = SevenNetInference(model_name="SevenNet-0")
    result = calc.optimize(atoms, fmax=0.01)
    
except ModelNotFoundError as e:
    print(f"模型错误: {e}")
    
except DeviceError as e:
    print(f"设备错误: {e}")
    
except StructureError as e:
    print(f"结构错误: {e}")
    
except ConvergenceError as e:
    print(f"未收敛: {e}")
    print("尝试增加 max_steps 或放宽 fmax")
    
except CalculationError as e:
    print(f"计算失败: {e}")
    
except SevenNetError as e:
    print(f"SevenNet 错误: {e}")
```

---

## 附录

### A. 返回值类型汇总

| 方法 | 返回类型 | 主要键 |
|------|----------|--------|
| `single_point` | Dict | energy, forces, stress, max_force |
| `optimize` | Dict | converged, atoms, final_energy, steps |
| `run_md` | Atoms | 最终结构 |
| `phonon` | Dict | frequency_points, total_dos, thermal |
| `bulk_modulus` | Dict | bulk_modulus, v0, e0 |
| `elastic_constants` | Dict | elastic_tensor, bulk_modulus |

### B. 模型信息

| 模型 | 参数量 | 训练集 | 推荐场景 |
|------|--------|--------|----------|
| SevenNet-0 | ~2M | MPtrj | 通用计算 |
| SevenNet-0-22May2024 | ~2M | MPtrj | 最新版本 |

### C. 性能参考

| 任务 | 100 atoms | 500 atoms | 1000 atoms |
|------|-----------|-----------|------------|
| 单点 (GPU) | ~3 ms | ~12 ms | ~30 ms |
| 单点 (CPU) | ~30 ms | ~120 ms | ~300 ms |
| 优化 | ~1 s | ~5 s | ~15 s |
| MD (1000步) | ~3 s | ~12 s | ~30 s |

### D. 常见问题

**Q: 如何选择模型?**  
A: 默认使用 `SevenNet-0`,适合大多数场景。

**Q: GPU 内存不足怎么办?**  
A: 减少 `max_neighbors` 或使用 CPU。

**Q: 优化不收敛?**  
A: 增加 `max_steps`,放宽 `fmax`,或尝试 FIRE 优化器。

**Q: 如何加速批量计算?**  
A: 使用 GPU,启用混合精度,或并行处理。

---

**最后更新**: 2026-01-07  
**版本**: 0.1.0  
**许可**: MIT License  
**联系**: GitHub Issues
