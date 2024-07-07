# Dependencies

- `setuptools` version 42 or later
- `pybind11` 2.6.1 or newer
- `numpy`
- [CUDA Toolkits](https://developer.nvidia.com/cuda-toolkit) for CUDA supports

# Installation


CPU version (Install the package without CUDA support)

```bash
pip install .
```

CUDA version

```bash
export CUDA_HOME=...
export USE_CUDA=1
pip install .
```



# Solving Quadratic Programming Problems

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

# Example

- **Python (CPU):** [./example/cpu.ipynb](./example/cpu.ipynb)
- **Python (CUDA):** [./example/cuda.ipynb](./example/cuda.ipynb)
- **C++ (CPU):** [./c/core/test/test.cpp](./c/core/test/test.cpp)
- **C++ (CUDA):** [./c/core/test/test.cu](./c/core/test/test.cu)
