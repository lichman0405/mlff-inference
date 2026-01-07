# SevenNet Inference 安装指南

> **SevenNet**: MOFSimBench 排名 **#4** 的等变图神经网络力场  
> **开发团队**: KAIST (韩国科学技术院)  
> **特色**: 力预测精度高、多元素支持、计算效率优异

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
- **内存**: 8GB RAM (推荐 16GB+)
- **磁盘**: 5GB 可用空间

### GPU 版本额外要求
- **GPU**: NVIDIA GPU (计算能力 >= 6.0)
- **CUDA**: 11.8 或 12.1+
- **GPU 显存**: >= 8GB (推荐 >= 16GB)

---

## 2. 安装方法

### 方法 1: Pip 安装 (推荐)

#### CPU 版本
```bash
# 创建虚拟环境
conda create -n sevennet-cpu python=3.10
conda activate sevennet-cpu

# 安装 PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装 SevenNet 和依赖
pip install sevenn ase phonopy pymatgen

# 安装本推理包
cd SevenNet_inference
pip install -r requirements-cpu.txt
```

#### GPU 版本
```bash
# 创建虚拟环境
conda create -n sevennet-gpu python=3.10
conda activate sevennet-gpu

# 安装 PyTorch (GPU) - 根据 CUDA 版本选择
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 SevenNet 和依赖
pip install sevenn ase phonopy pymatgen

# 安装本推理包
cd SevenNet_inference
pip install -r requirements-gpu.txt
```

### 方法 2: Conda 完整环境

```bash
# 创建环境并安装所有依赖
conda create -n sevennet python=3.10
conda activate sevennet

# 安装 PyTorch (GPU)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# 或安装 PyTorch (CPU)
conda install pytorch cpuonly -c pytorch

# 安装其他依赖
pip install sevenn ase phonopy pymatgen scipy h5py matplotlib pandas tqdm pyyaml
```

### 方法 3: 开发者安装 (从源码)

```bash
# 克隆仓库
git clone https://github.com/your-org/sevennet-inference
cd sevennet-inference/sevennet-inference

# 安装开发模式
pip install -e ".[dev]"
```

---

## 3. 验证安装

### 3.1 检查 Python 包

```bash
# 检查 sevenn 版本
python -c "import sevenn; print(sevenn.__version__)"

# 检查推理包
python -c "from sevennet_inference import SevenNetInference; print('OK')"

# 检查命令行工具
sevennet-infer --help
```

### 3.2 运行测试

```bash
# 基础功能测试
python -m pytest tests/test_install.py -v

# 完整测试套件
python -m pytest tests/ -v
```

### 3.3 GPU 功能测试

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
```

---

## 4. GPU 配置

### 4.1 选择 GPU 设备

```python
from sevennet_inference import SevenNetInference

# 自动检测 (优先 GPU)
calc = SevenNetInference(device="auto")

# 强制使用 GPU 0
calc = SevenNetInference(device="cuda:0")

# 强制使用 CPU
calc = SevenNetInference(device="cpu")
```

### 4.2 多 GPU 环境

```python
import torch

# 查看所有可用 GPU
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 使用特定 GPU
calc = SevenNetInference(device="cuda:1")
```

### 4.3 内存优化

```python
# 使用混合精度 (节省显存)
calc = SevenNetInference(
    device="cuda",
    precision="float16"  # 或 "bfloat16"
)

# 批处理大小调整
calc = SevenNetInference(
    device="cuda",
    batch_size=32  # 根据显存调整
)
```

---

## 5. 常见问题

### 5.1 SevenNet 安装失败

**问题**: `pip install sevenn` 失败

**解决方法**:
```bash
# 尝试从 GitHub 安装
pip install git+https://github.com/MDIL-SNU/SevenNet.git

# 或手动克隆安装
git clone https://github.com/MDIL-SNU/SevenNet.git
cd SevenNet
pip install -e .
```

### 5.2 CUDA 内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方法**:
```python
# 1. 减小批处理大小
calc = SevenNetInference(batch_size=16)

# 2. 使用混合精度
calc = SevenNetInference(precision="float16")

# 3. 清理 GPU 缓存
import torch
torch.cuda.empty_cache()

# 4. 使用 CPU
calc = SevenNetInference(device="cpu")
```

### 5.3 模型文件未找到

**问题**: `FileNotFoundError: Model checkpoint not found`

**解决方法**:
```python
# 指定模型路径
calc = SevenNetInference(
    model_path="/path/to/sevennet_model.pt"
)

# 或下载预训练模型
from sevennet_inference.utils import download_pretrained_model
model_path = download_pretrained_model("SevenNet-0")
```

### 5.4 导入错误

**问题**: `ModuleNotFoundError: No module named 'sevenn'`

**解决方法**:
```bash
# 重新安装依赖
pip install --upgrade sevenn ase phonopy

# 检查环境
conda list | grep sevenn
```

### 5.5 ASE 版本不兼容

**问题**: ASE 版本冲突

**解决方法**:
```bash
# 安装兼容版本
pip install "ase>=3.22.0,<3.24.0"
```

---

## 6. 依赖说明

### 核心依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `sevenn` | >= 0.9.0 | SevenNet 模型核心 |
| `torch` | >= 2.0.0 | 深度学习框架 |
| `ase` | >= 3.22.0 | 原子模拟环境 |
| `numpy` | >= 1.21.0, < 2.0 | 数值计算 |
| `scipy` | >= 1.7.0 | 科学计算 |

### 可选依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| `phonopy` | >= 2.20.0 | 声子计算 |
| `pymatgen` | >= 2023.0.0 | 材料结构分析 |
| `spglib` | >= 2.0.0 | 空间群分析 |
| `matplotlib` | >= 3.5.0 | 可视化 |
| `pandas` | latest | 数据处理 |

### 开发依赖

```bash
# 安装开发工具
pip install pytest pytest-cov black flake8 mypy
```

---

## 7. 性能优化建议

### 7.1 CPU 优化
```bash
# 设置 OpenMP 线程数
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### 7.2 GPU 优化
```python
# 启用 TF32 (A100 GPU)
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 使用 cudnn benchmark
torch.backends.cudnn.benchmark = True
```

---

## 8. 卸载

```bash
# 卸载推理包
pip uninstall sevennet-inference

# 卸载 SevenNet
pip uninstall sevenn

# 删除 conda 环境
conda deactivate
conda remove -n sevennet-gpu --all
```

---

## 9. 获取帮助

- **文档**: [SevenNet_inference_tasks.md](SevenNet_inference_tasks.md)
- **API 参考**: [SevenNet_inference_API_reference.md](SevenNet_inference_API_reference.md)
- **示例代码**: `sevennet-inference/examples/`
- **问题反馈**: GitHub Issues

---

## 10. 更新日志

### v0.1.0 (2026-01-07)
- 初始版本发布
- 支持 SevenNet-0 模型
- 实现单点计算、结构优化、MD 模拟
- 支持 CPU 和 GPU 加速
