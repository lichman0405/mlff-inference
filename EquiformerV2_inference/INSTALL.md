# EquiformerV2 Inference 安装指南

> **EquiformerV2**: 基于等变 Transformer 的高精度力场模型  
> **开发团队**: Meta AI / UC Berkeley - Open Catalyst Project  
> **特色**: S2EF 精度优异、E(3) 等变架构、大规模材料预训练

---

## 📋 目录

1. [系统要求](#1-系统要求)
2. [安装方法](#2-安装方法)
3. [验证安装](#3-验证安装)
4. [GPU 配置](#4-gpu-配置)
5. [常见问题](#5-常见问题)
6. [依赖说明](#6-依赖说明)

---

## 1. 系统要求

### 最低配置
- **操作系统**: Linux, macOS, Windows (WSL2)
- **Python**: 3.9 - 3.11 (推荐 3.10)
- **内存**: 12GB RAM (推荐 16GB+)
- **磁盘**: 8GB 可用空间

### GPU 版本额外要求
- **GPU**: NVIDIA GPU (计算能力 >= 6.0)
- **CUDA**: 11.8 或 12.1+
- **GPU 显存**: >= 12GB (推荐 >= 16GB)
- **驱动**: NVIDIA Driver >= 450.80.02

---

## 2. 安装方法

### 方法 1: Pip 安装 (推荐)

#### CPU 版本
```bash
# 创建虚拟环境
conda create -n equiformerv2-cpu python=3.10
conda activate equiformerv2-cpu

# 安装 PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 EquiformerV2 核心依赖
pip install equiformer-v2 e3nn

# 安装原子模拟环境
pip install ase phonopy pymatgen

# 安装本推理包
cd EquiformerV2_inference
pip install -r requirements-cpu.txt
```

#### GPU 版本
```bash
# 创建虚拟环境
conda create -n equiformerv2-gpu python=3.10
conda activate equiformerv2-gpu

# 安装 PyTorch (GPU) - 根据 CUDA 版本选择
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 torch-geometric 及其依赖
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric

# 安装 EquiformerV2 核心依赖
pip install equiformer-v2 e3nn

# 安装原子模拟环境
pip install ase phonopy pymatgen spglib

# 安装本推理包
cd EquiformerV2_inference
pip install -r requirements-gpu.txt
```

### 方法 2: Conda 完整环境

```bash
# 创建环境并安装所有依赖
conda create -n equiformerv2 python=3.10
conda activate equiformerv2

# 安装 PyTorch (GPU)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# 或安装 PyTorch (CPU)
conda install pytorch cpuonly -c pytorch

# 安装 PyTorch Geometric
conda install pyg -c pyg

# 安装 E3NN
pip install e3nn

# 安装其他依赖
pip install equiformer-v2 ase phonopy pymatgen scipy h5py matplotlib pandas tqdm pyyaml spglib
```

### 方法 3: 开发者安装 (从源码)

```bash
# 克隆仓库
cd EquiformerV2_inference/equiformerv2-inference

# 安装开发模式
pip install -e ".[dev]"

# 安装测试依赖
pip install pytest pytest-cov
```

---

## 3. 验证安装

### 3.1 检查 Python 包

```bash
# 检查 EquiformerV2 版本
python -c "import equiformer_v2; print(equiformer_v2.__version__)"

# 检查 E3NN
python -c "import e3nn; print(f'E3NN version: {e3nn.__version__}')"

# 检查 torch-geometric (GPU 版本)
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"

# 检查推理包
python -c "from equiformerv2_inference import EquiformerV2Inference; print('EquiformerV2 Inference OK')"

# 检查命令行工具
equiformerv2-infer --help
```

### 3.2 运行测试

```bash
# 基础功能测试
cd EquiformerV2_inference/equiformerv2-inference
python -m pytest tests/test_install.py -v

# 完整测试套件
python -m pytest tests/ -v

# 运行示例脚本
python examples/01_single_point.py
```

### 3.3 GPU 功能测试

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
```

### 3.4 E3NN 功能测试

```python
import torch
from e3nn import o3

# 测试球谐函数
irreps = o3.Irreps("1e + 2e")
print(f"Irreps: {irreps}")

# 测试旋转等变性
x = torch.randn(10, irreps.dim)
print(f"E3NN test passed! Tensor shape: {x.shape}")
```

---

## 4. GPU 配置

### 4.1 选择 GPU 设备

```python
from equiformerv2_inference import EquiformerV2Inference

# 自动检测 (优先 GPU)
calc = EquiformerV2Inference(device="auto")

# 强制使用 GPU 0
calc = EquiformerV2Inference(device="cuda:0")

# 强制使用 CPU
calc = EquiformerV2Inference(device="cpu")
```

### 4.2 多 GPU 环境

```python
import torch

# 查看所有可用 GPU
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")

# 使用特定 GPU
calc = EquiformerV2Inference(device="cuda:1")

# 环境变量控制可见 GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"  # 只使用 GPU 0 和 2
```

### 4.3 内存优化

```python
# 使用混合精度 (节省显存，略微降低精度)
calc = EquiformerV2Inference(
    device="cuda",
    precision="float16"  # 或 "bfloat16" (A100)
)

# 批处理大小调整
calc = EquiformerV2Inference(
    device="cuda",
    batch_size=16  # 根据显存调整 (默认 32)
)

# 启用梯度检查点 (训练用，推理通常不需要)
calc = EquiformerV2Inference(
    device="cuda",
    gradient_checkpointing=True
)
```

### 4.4 性能优化

```python
import torch

# 启用 TF32 (Ampere 架构: A100, RTX 30 系列)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 启用 cuDNN benchmark (固定输入大小时)
torch.backends.cudnn.benchmark = True

# 设置 cuDNN deterministic (牺牲性能保证可重复性)
torch.backends.cudnn.deterministic = False  # 推理时设为 False
```

| CUDA 版本 | PyTorch 版本 | 推荐使用 |
|-----------|-------------|---------|
| 11.8 | 2.0.0+ | ✅ 稳定 |
| 12.1 | 2.1.0+ | ✅ 推荐 |
| 12.4 | 2.3.0+ | ⚠️ 测试中 |

检查兼容性：
```bash
# 检查 CUDA 版本
nvcc --version

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"
```

---

## 5. 常见问题

### 5.1 安装问题

#### Q1: PyG 扩展安装失败

**问题**: `torch-scatter`, `torch-sparse` 等安装失败

**解决方案**:
```bash
# 方法 1: 使用预编译轮子
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 方法 2: 使用 conda
conda install pyg -c pyg

# 方法 3: 从源码编译
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
pip install torch-scatter torch-sparse --no-cache-dir
```

#### Q2: ImportError: cannot import name 'EquiformerV2Inference'

**问题**: 推理包未正确安装

**解决方案**:
```bash
# 重新安装推理包
cd equiformerv2-inference
pip install -e . --force-reinstall

# 检查 Python 路径
python -c "import sys; print(sys.path)"
```

#### Q3: E3NN 安装失败

**问题**: `pip install e3nn` 失败或版本不兼容

**解决方案**:
```bash
# 安装特定版本
pip install "e3nn>=0.5.0,<0.6.0"

# 或从 conda-forge 安装
conda install -c conda-forge e3nn

# 或从源码安装
pip install git+https://github.com/e3nn/e3nn.git
```

#### Q4: NumPy 版本冲突

**问题**: `numpy>=2.0` 与某些包不兼容

**解决方案**:
```bash
# 降级到 NumPy 1.x
pip install "numpy>=1.21.0,<2.0.0"

# 或使用兼容的 NumPy 2.0
pip install --upgrade numpy scipy
```

#### Q5: ASE 版本不兼容

**问题**: ASE 接口变化导致错误

**解决方案**:
```bash
# 安装推荐版本
pip install "ase>=3.22.0,<3.24.0"

# 或更新到最新
pip install --upgrade ase
```

### 5.2 运行时问题

#### Q6: 模型加载错误

**问题**: `FileNotFoundError: Model checkpoint not found`

**解决方案**:
```python
# 方法 1: 显式指定模型路径
calc = EquiformerV2Inference(
    model_path="/path/to/equiformer_v2_checkpoint.pt"
)

# 方法 2: 下载预训练模型
from equiformerv2_inference.utils import download_pretrained_model
model_path = download_pretrained_model("EquiformerV2-31M-S2EF")
calc = EquiformerV2Inference(model_path=model_path)

# 方法 3: 从 OCP 模型库下载
# 访问: https://github.com/Open-Catalyst-Project/ocp
```

#### Q7: CUDA out of memory

**问题**: GPU 显存不足

**解决方案**:
```python
# 1. 减小批处理大小
calc = EquiformerV2Inference(batch_size=8)

# 2. 使用混合精度
calc = EquiformerV2Inference(precision="float16")

# 3. 清理 GPU 缓存
import torch
torch.cuda.empty_cache()

# 4. 限制 PyTorch 内存分配
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# 5. 切换到 CPU
calc = EquiformerV2Inference(device="cpu")
```

#### Q8: 导入错误

**问题**: `ModuleNotFoundError: No module named 'equiformer_v2'` 或 `e3nn`

**解决方案**:
```bash
# 重新安装依赖
pip install --upgrade equiformer-v2 e3nn torch-geometric

# 检查环境
conda list | grep -E "equiformer|e3nn|torch"

# 检查 Python 路径
python -c "import sys; print('\n'.join(sys.path))"

# 重新安装推理包
cd EquiformerV2_inference/equiformerv2-inference
pip install -e .
```

#### Q9: Phonopy 计算失败

**问题**: 声子计算报错

**解决方案**:
```bash
# 确保安装完整依赖
pip install phonopy spglib h5py pyyaml matplotlib

# 检查版本
python -c "import phonopy; print(phonopy.__version__)"
python -c "import spglib; print(spglib.__version__)"
```

### 5.3 性能问题

**问题**: 分子动力学模拟中结构崩溃

**原因**: EquiformerV2 使用非保守力 (直接预测力而非从能量梯度计算)

**解决方案**:
```python
# 方案 1: 减小时间步长
md_result = calc.run_md(
    atoms,
    ensemble="nvt",
    temperature=300,
    timestep=0.5,  # 降低到 0.5 fs (默认 1.0 fs)
    steps=50000
)

# 方案 2: 使用更严格的温度控制
md_result = calc.run_md(
    atoms,
    ensemble="nvt",
    temperature=300,
    thermostat_time_constant=50,  # 减小耦合常数
    timestep=0.5
)

# 方案 3: 先优化结构
opt_result = calc.optimize(atoms, fmax=0.01)
optimized_atoms = opt_result['atoms']
md_result = calc.run_md(optimized_atoms, ...)
```

#### Q6: 力预测不准确

**问题**: 力的 MAE 较高

**原因**: EquiformerV2 的非保守力设计

**说明**:
- EquiformerV2 直接预测力 (非保守)，计算效率高但物理一致性较差
- 对于需要高精度保守力的任务，推荐使用 MACE 或 eSEN

```python
# 如需保守力，切换到其他模型
from mace.calculators import mace_mp
calc = mace_mp(model="medium", device="cuda")  # MACE 使用保守力
```

### 5.3 性能问题

#### Q10: 计算速度慢

**解决方案**:
```python
# 1. 确保使用 GPU
calc = EquiformerV2Inference(device="cuda")

# 2. 使用小模型
calc = EquiformerV2Inference(model_name="EquiformerV2-31M-S2EF")  # 而非 153M

# 3. 增加批量大小 (高通量)
results = calc.batch_inference(atoms_list, batch_size=8)

# 4. 使用混合精度
calc = EquiformerV2Inference(
    device="cuda",
    precision="float16"
)
```

#### Q11: MD 模拟不稳定 / 能量爆炸

**问题**: 分子动力学模拟中结构崩溃

**原因**: EquiformerV2 使用非保守力 (直接预测力而非从能量梯度计算)

**解决方案**:
```python
# 方案 1: 减小时间步长
md_result = calc.run_md(
    atoms,
    ensemble="nvt",
    temperature=300,
    timestep=0.5,  # 降低到 0.5 fs (默认 1.0 fs)
    steps=50000
)

# 方案 2: 使用更严格的温度控制
md_result = calc.run_md(
    atoms,
    ensemble="nvt",
    temperature=300,
    thermostat_time_constant=50,  # 减小耦合常数
    timestep=0.5
)

# 方案 3: 先优化结构
opt_result = calc.optimize(atoms, fmax=0.01)
optimized_atoms = opt_result['atoms']
md_result = calc.run_md(optimized_atoms, ...)
```

#### Q12: 多进程计算问题

**解决方案**:
```python
# 避免在多进程中共享 CUDA 张量
from multiprocessing import Pool
from ase.io import read

def worker(structure_file):
    # 每个进程创建独立的计算器
    calc = EquiformerV2Inference(
        device="cpu"  # 多进程建议使用 CPU
    )
    atoms = read(structure_file)
    return calc.single_point(atoms)

with Pool(processes=4) as pool:
    results = pool.map(worker, structure_files)
```

---

## 6. 依赖说明

### 6.1 核心依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `equiformer-v2` | >= 0.1.0 | EquiformerV2 模型核心 |
| `e3nn` | >= 0.5.0 | E(3) 等变神经网络操作 |
| `torch` | >= 2.0.0 | 深度学习框架 |
| `torch-geometric` | >= 2.3.0 | 图神经网络库 (GPU 版本) |
| `ase` | >= 3.22.0 | 原子模拟环境 |
| `numpy` | >= 1.21.0, < 2.0 | 数值计算 |
| `scipy` | >= 1.7.0 | 科学计算 |

### 6.2 计算任务依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `phonopy` | >= 2.20.0 | 声子谱计算 |
| `pymatgen` | >= 2023.0.0 | 材料结构分析与处理 |
| `spglib` | >= 2.0.0 | 空间群对称性分析 |

### 6.3 工具依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `h5py` | latest | HDF5 文件读写 |
| `matplotlib` | >= 3.5.0 | 数据可视化 |
| `pandas` | latest | 数据分析与处理 |
| `tqdm` | latest | 进度条显示 |
| `pyyaml` | latest | YAML 配置文件 |
| `prettytable` | latest | 表格格式输出 |

### 6.4 Torch-Geometric 扩展 (GPU 版本)

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `torch-scatter` | 与 PyTorch 匹配 | 分散/聚集操作 |
| `torch-sparse` | 与 PyTorch 匹配 | 稀疏张量操作 |
| `torch-cluster` | 与 PyTorch 匹配 | 聚类算法 |
| `torch-spline-conv` | 与 PyTorch 匹配 | 样条卷积 |

### 6.5 开发依赖

```bash
# 测试工具
pip install pytest pytest-cov pytest-xdist

# 代码质量
pip install black flake8 mypy isort

# 文档生成
pip install sphinx sphinx-rtd-theme
```

---

## 7. 性能优化建议

### 7.1 CPU 优化
```bash
# 设置 OpenMP 线程数 (根据 CPU 核心数调整)
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# 使用 Intel MKL (更快的线性代数)
conda install mkl mkl-include
```

### 7.2 GPU 优化
```python
import torch

# 启用 TF32 (Ampere 架构: A100, RTX 3090)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 启用 cuDNN benchmark (输入大小固定时)
torch.backends.cudnn.benchmark = True

# 预分配 GPU 内存 (减少碎片化)
torch.cuda.empty_cache()
```

### 7.3 批处理优化
```python
# 根据系统资源调整批处理大小
from equiformerv2_inference import EquiformerV2Inference

# 小 GPU (8GB)
calc = EquiformerV2Inference(device="cuda", batch_size=8)

# 中等 GPU (16GB)
calc = EquiformerV2Inference(device="cuda", batch_size=32)

# 大 GPU (24GB+)
calc = EquiformerV2Inference(device="cuda", batch_size=64)
```

---

## 8. 卸载

```bash
# 卸载推理包
pip uninstall equiformerv2-inference

# 卸载核心依赖
pip uninstall equiformer-v2 e3nn torch-geometric

# 删除 conda 环境
conda deactivate
conda remove -n equiformerv2-gpu --all
# 或
conda remove -n equiformerv2-cpu --all
```

---

## 9. 获取帮助

- **快速入门**: [QUICKSTART.md](equiformerv2-inference/QUICKSTART.md)
- **详细安装**: [INSTALL_GUIDE.md](equiformerv2-inference/INSTALL_GUIDE.md)
- **示例代码**: [equiformerv2-inference/examples/](equiformerv2-inference/examples/)
- **测试脚本**: [equiformerv2-inference/tests/](equiformerv2-inference/tests/)
- **OCP 文档**: https://github.com/Open-Catalyst-Project/ocp
- **E3NN 文档**: https://docs.e3nn.org/

---

## 10. 架构说明

### EquiformerV2 特性
- **等变 Transformer**: 利用 E(3) 等变注意力机制
- **原子级预测**: 能量、力、应力张量
- **大规模预训练**: 在 OC20/OC22 数据集训练
- **高效推理**: 支持 batch 推理和 GPU 加速

### E3NN 核心概念
- **不可约表示 (Irreps)**: 球谐函数基
- **张量积**: 等变特征融合
- **旋转等变性**: 保持物理对称性

---

## 11. 更新日志

### v0.1.0 (2026-01-07)
- 初始版本发布
- 支持 EquiformerV2 预训练模型
- 实现单点能量/力计算
- 支持结构优化 (BFGS, LBFGS)
- 支持分子动力学 (NVE, NVT, NPT)
- 支持声子谱计算
- 支持体模量计算
- CPU/GPU 双模式
