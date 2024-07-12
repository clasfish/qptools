# Fast quadratic programming tools (qptools)

[中文](README.zh-CN.md) | [English](README.md)

## Platforms

- **Windows**: To Do
- **Linux**: Supports CPU and CUDA versions
- **MacOS**: Supports CPU version

## Dependencies

- `setuptools`
- `pybind11`
- `numpy`
- [CUDA Toolkits](https://developer.nvidia.com/cuda-toolkit) for CUDA supports

## Installation

### CPU version

Install the package without CUDA support

```bash
pip install qptools
```

### CUDA version

```bash
export CUDA_HOME=(your cuda path)
export USE_CUDA=1
pip install qptools
```


## Solving Quadratic Programming Problems

### `qp1`: Basic Quadratic Programming

$$
\min_{x} \frac12 x^TPx+q^Tx \\
\text{s.t. } lb \leq x \leq rb, \\
Gx \leq h
$$

where $P$ is a $n\times n$ positive definite matrix, $q$ is an $n$-dimensional vector, $G$ is a $g\times n$ matrix, $h$ is a $g$-dimensional vector, and $lb,rb$ are $n$-dimensional vectors defining the lower and upper bounds for variable $x$.

### `qp2`:  Quadratic Programming with Standardized Variables

$$
\min_{x} \frac12 x^TPx+q^Tx \\
\text{s.t. } lb \leq \frac{x}{\text{sum}(x)} \leq rb, \\
Gx \leq h
$$

where $P$ is a $(n,n)$ positive definite matrix, $q$ is an $n$-dimensional vector, $G$ is a $(g,n)$ matrix, $h$ is a $g$-dimensional vector, and $lb,rb$ are $n$-dimensional vectors defining the lower and upper bounds for standardized variable $\frac{x}{\sum(x)}$.

## Example

- **Python (CPU):** [./example/cpu.ipynb](./example/cpu.ipynb)
- **Python (CUDA):** [./example/cuda.ipynb](./example/cuda.ipynb)
- **C++ (CPU):** [./c/core/test/test.cpp](./c/core/test/test.cpp)
- **C++ (CUDA):** [./c/core/test/test.cu](./c/core/test/test.cu)

## References

1. Nesterov, Y., & Todd, M. J. (1997). Self-scaled barriers and interior-point methods for convex programming. Mathematics of Operations Research, 22(1), 1-42.

2. Nesterov, Y., & Todd, M. J. (1998). Primal-dual interior-point methods for self-scaled cones. SIAM Journal on Optimization, 8(2), 324-364.

3. Mehrotra, S. (1992). On the implementation of a primal-dual interior point method. SIAM Journal on Optimization, 2(4), 575-601.