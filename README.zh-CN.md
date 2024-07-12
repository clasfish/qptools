[中文](README.zh-CN.md)
[English](README.md)

# 依赖

- `setuptools`
- `pybind11`
- `numpy`
- [CUDA Toolkits](https://developer.nvidia.com/cuda-toolkit) for CUDA supports

# 安装

### CPU 版本

```bash
pip install qptools
```

CUDA version

```bash
export CUDA_HOME=(your cuda path)
export USE_CUDA=1
pip install qptools
```

# 解决二次优化问题

### `qp1`: 基础二次优化问题

$$
\min_{x} \frac12 x^TPx+q^Tx \\
\text{s.t. } lb \leq x \leq rb, \\
Gx \leq h
$$

其中 $P$ 是一个 $n\times n$ 的正定矩阵, $q$ 是一个 $n$ 维向量, $G$ 是一个 $g\times n$ 的矩阵, $h$ 是一个 $n$ 维向量, $lb$ 和 $rb$ 是 $n$ 维向量

### `qp2`:  二次优化问题(归一化的变量)

$$
\min_{x} \frac12 x^TPx+q^Tx \\
\text{s.t. } lb \leq \frac{x}{\text{sum}(x)} \leq rb, \\
Gx \leq h
$$

其中 $P$ 是一个 $n\times n$ 的正定矩阵, $q$ 是一个 $n$ 维向量, $G$ 是一个 $g\times n$ 的矩阵, $h$ 是一个 $n$ 维向量, $lb$ 和 $rb$ 是 $n$ 维向量

# 例子

- **Python (CPU):** [./example/cpu.ipynb](./example/cpu.ipynb)
- **Python (CUDA):** [./example/cuda.ipynb](./example/cuda.ipynb)
- **C++ (CPU):** [./c/core/test/test.cpp](./c/core/test/test.cpp)
- **C++ (CUDA):** [./c/core/test/test.cu](./c/core/test/test.cu)


# 参考文献

1. Nesterov, Y., & Todd, M. J. (1997). Self-scaled barriers and interior-point methods for convex programming. Mathematics of Operations Research, 22(1), 1-42.

2. Nesterov, Y., & Todd, M. J. (1998). Primal-dual interior-point methods for self-scaled cones. SIAM Journal on Optimization, 8(2), 324-364.

3. Mehrotra, S. (1992). On the implementation of a primal-dual interior point method. SIAM Journal on Optimization, 2(4), 575-601.