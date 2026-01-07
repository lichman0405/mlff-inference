# Orb 推理环境安装指南

> **文档版本**: v1.0  
> **最后更新**: 2026年1月7日  
> **适用模型**: Orb v2, Orb v3

本文档提供Orb系列模型推理环境的完整安装指南，包括CPU和GPU版本。

---

## 目录

1. [环境要求](#1-环境要求)
2. [安装步骤](#2-安装步骤)
   - [2.1 CPU版本安装](#21-cpu版本安装)
   - [2.2 GPU版本安装](#22-gpu版本安装)
3. [环境测试](#3-环境测试)
4. [故障排除](#4-故障排除)
5. [版本选择指南](#5-版本选择指南)

---

## 1. 环境要求

### 1.1 基础要求

| 组件 | CPU版本 | GPU版本 |
|------|---------|---------|
| **Python** | 3.8 - 3.11 | 3.8 - 3.11 |
| **操作系统** | Linux, macOS, Windows | Linux, Windows |
| **内存** | ≥ 8 GB | ≥ 16 GB |
| **存储空间** | ≥ 10 GB | ≥ 20 GB |

### 1.2 GPU要求（仅GPU版本）

| 组件 | 要求 |
|------|------|
| **GPU** | NVIDIA GPU（支持CUDA） |
| **显存** | ≥ 6 GB（推荐 ≥ 12 GB） |
| **CUDA** | 11.8 或 12.1+ |
| **驱动** | 与CUDA版本匹配 |

---

## 2. 安装步骤

### 2.1 CPU版本安装

#### 步骤1：创建Conda环境

```bash
# 创建新环境
conda create -n orb-cpu python=3.10
conda activate orb-cpu
```

#### 步骤2：安装核心依赖

```bash
# 安装PyTorch（CPU版本）
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# 安装Orb模型和相关库
pip install orb-models
pip install ase phonopy numpy matplotlib
```

**或使用requirements-cpu.txt一键安装**：

```bash
pip install -r requirements-cpu.txt
```

#### 步骤3：验证安装

```python
python -c "from orb_models.forcefield import pretrained; print('Orb models installed successfully!')"
```

---

### 2.2 GPU版本安装

#### 步骤1：创建Conda环境

```bash
# 创建新环境
conda create -n orb-gpu python=3.10
conda activate orb-gpu
```

#### 步骤2：安装PyTorch（GPU版本）

**CUDA 11.8**:
```bash
pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1+**:
```bash
pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 步骤3：安装Orb模型和相关库

```bash
pip install orb-models
pip install ase phonopy numpy matplotlib
```

**或使用requirements-gpu.txt一键安装**：

```bash
pip install -r requirements-gpu.txt
```

#### 步骤4：验证GPU支持

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

---

## 3. 环境测试

### 3.1 快速测试脚本

创建文件 `test_orb.py`：

```python
"""Orb环境测试脚本"""

import sys

def test_imports():
    """测试必需库的导入"""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False
    
    try:
        import ase
        print(f"  ✓ ASE {ase.__version__}")
    except ImportError as e:
        print(f"  ✗ ASE import failed: {e}")
        return False
    
    try:
        import phonopy
        print(f"  ✓ Phonopy {phonopy.__version__}")
    except ImportError as e:
        print(f"  ✗ Phonopy import failed: {e}")
        return False
    
    try:
        import orb_models
        print(f"  ✓ orb-models")
    except ImportError as e:
        print(f"  ✗ orb-models import failed: {e}")
        return False
    
    return True


def test_cuda():
    """测试CUDA支持"""
    print("\nTesting CUDA support...")
    
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA is available")
        print(f"    CUDA version: {torch.version.cuda}")
        print(f"    GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"  ℹ CUDA is not available (CPU-only mode)")
    
    return True


def test_orb_model():
    """测试Orb模型加载"""
    print("\nTesting Orb model loading...")
    
    try:
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        
        # 测试加载模型（CPU）
        device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
        print(f"  Loading orb-v3-omat model on {device}...")
        
        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=device,
            precision="float32-high"
        )
        calc = ORBCalculator(orbff, device=device)
        print(f"  ✓ Model loaded successfully on {device}")
        
        # 简单测试
        from ase.build import bulk
        atoms = bulk('Cu', 'fcc', a=3.58)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"  ✓ Test calculation: Cu bulk energy = {energy:.6f} eV")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("Orb Environment Test")
    print("="*60)
    
    results = []
    
    # 测试导入
    results.append(("Imports", test_imports()))
    
    # 测试CUDA
    results.append(("CUDA", test_cuda()))
    
    # 测试模型加载
    results.append(("Model Loading", test_orb_model()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✓ All tests passed! Environment is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

**运行测试**：

```bash
python test_orb.py
```

### 3.2 单独测试Orb计算

```python
from ase.build import bulk
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

# 加载模型
device = "cuda"  # 或 "cpu"
orbff = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high")
calc = ORBCalculator(orbff, device=device)

# 创建测试结构
atoms = bulk('Cu', 'fcc', a=3.58, cubic=True)
atoms.calc = calc

# 计算
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"Energy: {energy:.6f} eV")
print(f"Max force: {forces.max():.6f} eV/Å")
print("✓ Orb calculation successful!")
```

---

## 4. 故障排除

### 4.1 PyTorch版本问题

**问题**：安装的PyTorch版本与CUDA不匹配

**解决方案**：
```bash
# 卸载现有PyTorch
pip uninstall torch torchvision

# 根据CUDA版本重新安装
# CUDA 11.8
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

**重要**：
- ⚠️ **避免使用PyTorch 2.4.1**（与orb-models可能不兼容）
- ✅ **推荐使用PyTorch 2.3.1**

---

### 4.2 CUDA内存不足

**问题**：GPU显存溢出（Out of Memory）

**解决方案**：

1. **降低精度**：
```python
orbff = pretrained.orb_v3_conservative_inf_omat(
    device="cuda",
    precision="float32-high"  # 而非 "float32-highest" 或 "float64"
)
```

2. **减小体系规模**：
   - 使用更小的超胞
   - 减少结构优化时的并行任务

3. **清理GPU缓存**：
```python
import torch
torch.cuda.empty_cache()
```

4. **降级到CPU**：
```python
device = "cpu"  # 临时切换
```

---

### 4.3 orb-models安装失败

**问题**：`pip install orb-models` 失败

**解决方案**：

1. **更新pip**：
```bash
pip install --upgrade pip setuptools wheel
```

2. **从源码安装**：
```bash
git clone https://github.com/orbital-materials/orb-models.git
cd orb-models
pip install -e .
```

3. **检查网络**：
   - 使用镜像源（如清华源）
```bash
pip install orb-models -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 4.4 ASE版本冲突

**问题**：ASE版本过低，FrechetCellFilter不可用

**解决方案**：
```bash
pip install --upgrade ase>=3.23.0
```

**验证**：
```python
from ase.constraints import FrechetCellFilter  # 应成功导入
```

---

### 4.5 Phonopy计算报错

**问题**：声子计算时出现维度不匹配

**可能原因**：
- ASE Atoms与Phonopy Atoms转换问题
- 超胞矩阵设置错误

**解决方案**：

确保正确转换：
```python
from phonopy.structure.atoms import PhonopyAtoms

def ase_to_phonopy(atoms):
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,  # 注意使用.array
        positions=atoms.positions,
        masses=atoms.get_masses()
    )
```

---

## 5. 版本选择指南

### 5.1 Python版本

| Python版本 | 支持状态 | 推荐 |
|-----------|---------|------|
| 3.8 | ✅ 支持 | - |
| 3.9 | ✅ 支持 | - |
| **3.10** | ✅ 支持 | ⭐⭐⭐ **推荐** |
| 3.11 | ✅ 支持 | ⭐⭐ |
| 3.12 | ⚠️ 部分支持 | 不推荐（依赖库可能不兼容） |

### 5.2 PyTorch版本

| PyTorch版本 | 状态 | 说明 |
|------------|------|------|
| 2.0.x | ✅ 支持 | 最低支持版本 |
| 2.1.x | ✅ 支持 | 稳定 |
| 2.2.x | ✅ 支持 | 稳定 |
| **2.3.1** | ✅ 支持 | ⭐⭐⭐ **推荐** |
| 2.4.0 | ✅ 支持 | 可用 |
| 2.4.1 | ⚠️ 避免 | ❌ **可能不兼容orb-models** |

### 5.3 Orb模型版本选择

| 使用场景 | 推荐模型 | 理由 |
|---------|---------|------|
| **MOF推理任务** | orb-v3-omat | 性能最佳，热容预测第一 |
| **广泛材料类型** | orb-v3-mpa | 更大化学覆盖范围 |
| **需要D3校正** | orb-d3-v2 | 内置D3（但非保守力） |
| **快速测试** | orb-v3-omat (float32-high) | 速度与精度平衡 |
| **高精度研究** | orb-v3-omat (float64) | 最高精度（慢） |

**关键建议**：
- ✅ **优先使用 orb-v3 系列**（保守力，稳定性远优于v2）
- ⚠️ **避免v2模型用于MD和结构优化**（非保守力导致不稳定）
- ✅ **对于MOF，首选 orb-v3-omat**

---

## 6. 完整依赖列表

### 6.1 requirements-cpu.txt

```
# PyTorch (CPU)
torch==2.3.1
torchvision

# Orb Models
orb-models

# Atomic Simulation Environment
ase>=3.23.0

# Phonon calculations
phonopy>=2.20.0

# Numerical computing
numpy>=1.20.0

# Visualization (optional)
matplotlib>=3.0.0

# Jupyter (optional)
jupyter
ipython
```

### 6.2 requirements-gpu.txt

```
# PyTorch (GPU) - 根据CUDA版本选择
# CUDA 11.8
torch==2.3.1
torchvision
# 或 CUDA 12.1
# torch==2.3.1
# torchvision

# Orb Models
orb-models

# Atomic Simulation Environment
ase>=3.23.0

# Phonon calculations
phonopy>=2.20.0

# Numerical computing
numpy>=1.20.0

# Visualization (optional)
matplotlib>=3.0.0

# Jupyter (optional)
jupyter
ipython
```

---

## 7. 常见问题FAQ

**Q1: Orb和MACE有什么区别？**

A: 
- **Orb**：学习等变性（GNS架构），计算更快，适合大规模筛选
- **MACE**：预定义等变性（E(3)群），理论基础更强，精度略高
- **推荐**：两者都用，对比结果

**Q2: GPU加速效果明显吗？**

A: 
- 单点计算：GPU快10-50倍
- 结构优化：GPU快5-20倍
- MD模拟：GPU快20-100倍
- **结论**：GPU强烈推荐

**Q3: 可以在Windows上使用吗？**

A: 
- CPU版本：✅ 完全支持
- GPU版本：✅ 支持（需安装CUDA）
- 推荐使用WSL2或原生Linux以获得最佳性能

**Q4: 如何选择精度（precision）？**

A:
- `float32-high`: 快速，适合大规模筛选
- `float32-highest`: 平衡，**推荐用于生产**
- `float64`: 最慢，仅用于高精度基准

---

## 8. 相关链接

- **Orb Models GitHub**: https://github.com/orbital-materials/orb-models
- **官方文档**: https://docs.orbitalmaterials.com/
- **ASE文档**: https://wiki.fysik.dtu.dk/ase/
- **Phonopy文档**: https://phonopy.github.io/phonopy/
- **PyTorch安装**: https://pytorch.org/get-started/locally/

---

*如有问题，请参考 [GitHub Issues](https://github.com/orbital-materials/orb-models/issues) 或联系开发者*

*最后更新: 2026年1月7日*
