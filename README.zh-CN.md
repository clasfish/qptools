
# qptools

快速二次优化工具

[中文](README.zh-CN.md) | [English](README.md)

## 平台支持

- **Windows**: 待开发
- **Linux**: 支持 CPU 和 CUDA 版本
- **MacOS**: 支持 CPU 版本


## 依赖

- `setuptools`
- `pybind11`
- `numpy`
- [CUDA Toolkits](https://developer.nvidia.com/cuda-toolkit) (CUDA版本)

## 安装

### CPU 版本安装

```bash
pip install qptools
```

### CUDA 版本安装

确保你的系统已配置CUDA路径，并设置环境变量。

```bash
export CUDA_HOME=(your cuda path)
export USE_CUDA=1
pip install qptools
```

## 解决二次优化问题

### `qp1`: 基础二次优化问题

$$
\min_{x} \frac12 x^TPx+q^Tx \\
\text{s.t. } lb \leq x \leq rb, \\
Gx \leq h
$$

其中 $P$ 是一个 $n\times n$ 的正定矩阵, $q$ 是一个 $n$ 维向量, $G$ 是一个 $g\times n$ 的矩阵, $h$ 是一个 $n$ 维向量, $lb$ 和 $rb$ 分别是 $n$ 维的下界和上界向量。

### `qp2`:  归一化变量的二次优化问题

$$
\min_{x} \frac12 x^TPx+q^Tx \\
\text{s.t. } lb \leq \frac{x}{\text{sum}(x)} \leq rb, \\
Gx \leq h
$$

与`qp1`类似，但约束条件增加了对$x$向量元素的归一化要求。

## 例子

- **Python (CPU)**: 查看[./example/cpu.ipynb](./example/cpu.ipynb)
- **Python (CUDA)**: 查看[./example/cuda.ipynb](./example/cuda.ipynb)
- **C++ (CPU)**: 查看[./c/core/test/test.cpp](./c/core/test/test.cpp)
- **C++ (CUDA)**: 查看[./c/core/test/test.cu](./c/core/test/test.cu)

## 引用文献

1. Nesterov, Y., & Todd, M. J. (1997). Self-scaled barriers and interior-point methods for convex programming. Mathematics of Operations Research, 22(1), 1-42.

2. Nesterov, Y., & Todd, M. J. (1998). Primal-dual interior-point methods for self-scaled cones. SIAM Journal on Optimization, 8(2), 324-364.

3. Mehrotra, S. (1992). On the implementation of a primal-dual interior point method. SIAM Journal on Optimization, 2(4), 575-601.