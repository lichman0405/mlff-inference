# MACE Inference 环境配置指南

## 概述

本文档提供MACE模型推理环境的详细配置说明。提供**CPU版本**和**GPU版本**两种配置，请根据您的硬件条件选择。
**相关文档**：
- [MACE 推理任务文档](MACE_inference_tasks.md)：完整的推理任务代码示例
- [MACE API 接口参考](MACE_inference_API_reference.md)：详细的输入输出说明和版本验证
## 版本选择指南

| 版本 | 适用场景 | 优势 | 劣势 |
|------|----------|------|------|
| **CPU版本** | 小型结构测试、开发调试、无GPU环境 | 易于配置、兼容性好 | 计算速度慢 |
| **GPU版本** | 大型MOF结构、长时间MD、批量计算 | 计算速度快（10-100倍） | 需要NVIDIA GPU |

### 推理任务耗时对比（参考）

| 任务 | CPU (单核) | GPU (RTX 3090) | 加速比 |
|------|------------|----------------|--------|
| 单点能量计算 (200原子) | ~2s | ~0.05s | 40x |
| 结构优化 (200原子, 100步) | ~5min | ~10s | 30x |
| MD 10ps (200原子) | ~2h | ~3min | 40x |
| 声子计算 (200原子, 2x2x2超胞) | ~10h | ~15min | 40x |

---

## 系统要求

### CPU版本

| 项目 | 要求 |
|------|------|
| Python | >= 3.9, <= 3.11 (推荐 **3.10**) |
| PyTorch | >= 1.12, **推荐 2.3.x** |
| 内存 | >= 16GB RAM |
| CPU | 多核推荐（可并行计算） |

### GPU版本

| 项目 | 要求 |
|------|------|
| Python | >= 3.9, <= 3.11 (推荐 **3.10**) |
| PyTorch | >= 1.12, **推荐 2.3.x** |
| CUDA | 11.8 或 12.1 (推荐 **12.1**) |
| 内存 | >= 16GB RAM |
| GPU显存 | >= 8GB (推荐 >= 16GB用于大型MOF结构) |
| GPU | NVIDIA GPU (计算能力 >= 6.0) |

### PyTorch版本注意事项

| PyTorch版本 | 状态 | 备注 |
|-------------|------|------|
| 2.5.x | ✅ 支持 | 最新稳定版 |
| 2.4.1 | ❌ **不支持** | 已知兼容性问题 |
| 2.3.x | ✅ 支持 | 推荐 |
| 2.2.x | ✅ 支持 | float64训练支持 |
| 2.1.x | ⚠️ 部分支持 | float64训练不支持 |
| 1.12-2.0 | ⚠️ 老旧 | 功能可能受限 |

---

## CPU版本安装

### 方法一：Conda环境 (推荐)

```bash
# Step 1: 创建环境
conda create -n mace-inference-cpu python=3.10 -y
conda activate mace-inference-cpu

# Step 2: 安装CPU版PyTorch
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 3: 安装依赖
cd MACE_inference
pip install -r requirements-cpu.txt

# Step 4: 验证
python -c "from mace.calculators import mace_mp; calc = mace_mp(model='medium', device='cpu'); print('CPU版本安装成功!')"
```

### 方法二：pip虚拟环境

```bash
# 创建虚拟环境
python -m venv mace-cpu-env

# 激活环境 (Windows PowerShell)
.\mace-cpu-env\Scripts\Activate.ps1

# 激活环境 (Linux/Mac)
source mace-cpu-env/bin/activate

# 安装PyTorch和依赖
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

### CPU版本验证脚本

```python
# test_cpu.py
from mace.calculators import mace_mp
from ase.build import bulk

# 测试MACE (CPU)
calc = mace_mp(model="medium", device="cpu")
print("✓ MACE加载成功 (CPU)")

# 测试D3校正
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
d3 = TorchDFTD3Calculator(device="cpu", damping="bj", xc="pbe")
print("✓ D3校正加载成功 (CPU)")

# 测试计算
atoms = bulk('Cu', 'fcc', a=3.6)
atoms.calc = calc
energy = atoms.get_potential_energy()
print(f"✓ 能量计算成功: {energy:.4f} eV")

# 测试Phonopy
import phonopy
print(f"✓ Phonopy版本: {phonopy.__version__}")

print("\n=== CPU版本安装验证通过 ===")
```

---

## GPU版本安装

### 前置检查

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version
```

### 方法一：Conda环境 (推荐)

```bash
# Step 1: 创建环境
conda create -n mace-inference-gpu python=3.10 -y
conda activate mace-inference-gpu

# Step 2: 安装GPU版PyTorch (选择对应CUDA版本)
# CUDA 12.1
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 或 CUDA 11.8
# pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 3: 安装依赖
cd MACE_inference
pip install -r requirements-gpu.txt

# Step 4: 验证
python -c "from mace.calculators import mace_mp; calc = mace_mp(model='medium', device='cuda'); print('GPU版本安装成功!')"
```

### 方法二：pip虚拟环境

```bash
# 创建虚拟环境
python -m venv mace-gpu-env

# 激活环境 (Windows PowerShell)
.\mace-gpu-env\Scripts\Activate.ps1

# 激活环境 (Linux/Mac)
source mace-gpu-env/bin/activate

# 安装PyTorch (CUDA 12.1) 和依赖
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
```

### GPU版本验证脚本

```python
# test_gpu.py
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU设备: {torch.cuda.get_device_name(0)}")

from mace.calculators import mace_mp
from ase.build import bulk

# 测试MACE (GPU)
calc = mace_mp(model="medium", device="cuda")
print("✓ MACE加载成功 (GPU)")

# 测试D3校正
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
d3 = TorchDFTD3Calculator(device="cuda", damping="bj", xc="pbe")
print("✓ D3校正加载成功 (GPU)")

# 测试计算
atoms = bulk('Cu', 'fcc', a=3.6)
atoms.calc = calc
energy = atoms.get_potential_energy()
print(f"✓ 能量计算成功: {energy:.4f} eV")

# 测试Phonopy
import phonopy
print(f"✓ Phonopy版本: {phonopy.__version__}")

print("\n=== GPU版本安装验证通过 ===")
```

---

## Docker部署 (适用于集群/服务器)

### GPU版本Dockerfile

```dockerfile
# Dockerfile.gpu
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY . .
CMD ["python", "inference.py"]
```

### CPU版本Dockerfile

```dockerfile
# Dockerfile.cpu
FROM python:3.10-slim

WORKDIR /app

# 安装CPU版PyTorch
RUN pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements-cpu.txt .
RUN pip install --no-cache-dir -r requirements-cpu.txt

COPY . .
CMD ["python", "inference.py"]
```

### 构建和运行

```bash
# GPU版本
docker build -f Dockerfile.gpu -t mace-inference-gpu .
docker run --gpus all -v $(pwd)/data:/app/data mace-inference-gpu

# CPU版本
docker build -f Dockerfile.cpu -t mace-inference-cpu .
docker run -v $(pwd)/data:/app/data mace-inference-cpu
```

---

## GPU版本可选加速：cuEquivariance

MACE支持使用cuEquivariance库进行额外CUDA加速，可提供约2-3倍的速度提升。

### 安装cuEquivariance

```bash
# CUDA 12
pip install cuequivariance-torch>=0.2.0
pip install cuequivariance-ops-torch-cu12>=0.2.0

# CUDA 11
pip install cuequivariance-torch>=0.2.0
pip install cuequivariance-ops-torch-cu11>=0.2.0
```

### 使用cuEquivariance

```python
from mace.calculators import mace_mp

# 启用cuEquivariance加速
calc = mace_mp(
    model="medium", 
    device="cuda",
    enable_cueq=True  # 启用CUDA加速
)
```

---

## 常见问题排查

### 1. e3nn版本冲突

```
ERROR: Cannot install e3nn==0.4.4 because...
```

**解决方案**：MACE强制要求e3nn==0.4.4，确保先安装MACE再安装其他库：

```bash
pip install mace-torch
# 然后安装其他依赖
```

### 2. CUDA版本不匹配 (仅GPU版本)

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**解决方案**：检查PyTorch CUDA版本与系统CUDA版本是否匹配：

```bash
# 检查系统CUDA版本
nvcc --version

# 检查PyTorch CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

### 3. 内存不足 (OOM) (仅GPU版本)

```
RuntimeError: CUDA out of memory
```

**解决方案**：
- 减小supercell大小
- 使用`float32`代替`float64`
- 使用`small`模型代替`medium`

```python
calc = mace_mp(
    model="small",  # 使用更小的模型
    device="cuda",
    default_dtype="float32"  # 使用float32
)
```

### 4. torch-dftd安装失败

```bash
# 如果pip安装失败，尝试从源码安装
pip install git+https://github.com/pfnet-research/torch-dftd.git
```

### 5. CPU版本运行太慢

**解决方案**：
- 减小结构大小或supercell
- 使用`small`模型
- 减少MD步数进行测试
- 考虑升级到GPU版本

```python
# CPU优化设置
calc = mace_mp(
    model="small",  # 最小模型
    device="cpu",
    default_dtype="float32"  # float32更快
)
```

---

## 快速安装脚本

### Windows PowerShell - GPU版本

```powershell
# install_gpu.ps1
conda create -n mace-inference-gpu python=3.10 -y
conda activate mace-inference-gpu
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
python -c "from mace.calculators import mace_mp; calc = mace_mp(model='medium', device='cuda'); print('GPU版本OK')"
```

### Windows PowerShell - CPU版本

```powershell
# install_cpu.ps1
conda create -n mace-inference-cpu python=3.10 -y
conda activate mace-inference-cpu
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
python -c "from mace.calculators import mace_mp; calc = mace_mp(model='medium', device='cpu'); print('CPU版本OK')"
```

### Linux Bash - GPU版本

```bash
#!/bin/bash
# install_gpu.sh
conda create -n mace-inference-gpu python=3.10 -y
source activate mace-inference-gpu
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
python -c "from mace.calculators import mace_mp; calc = mace_mp(model='medium', device='cuda'); print('GPU版本OK')"
```

### Linux Bash - CPU版本

```bash
#!/bin/bash
# install_cpu.sh
conda create -n mace-inference-cpu python=3.10 -y
source activate mace-inference-cpu
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
python -c "from mace.calculators import mace_mp; calc = mace_mp(model='medium', device='cpu'); print('CPU版本OK')"
```

---

## 环境导出

安装完成后，建议导出环境以便复现：

```bash
# Conda环境导出
conda env export > environment.yml

# pip环境导出
pip freeze > requirements-lock.txt
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `requirements-cpu.txt` | CPU版本依赖 |
| `requirements-gpu.txt` | GPU版本依赖 |
| `INSTALL.md` | 本安装指南 |
| `MACE_inference_tasks.md` | 推理任务文档 |

---

*文档生成时间: 2026年1月7日*
