# Installation

## CPU version

Dependencies:

- setuptools
- pybind11
- numpy

Installation

```bash
pip install .
```

## CUDA version

Dependencies

- setuptools
- pybind11
- numpy
- [CUDA Toolkits](https://developer.nvidia.com/cuda-toolkit) (>=11.0)


Installation

```bash
pip install .
python setup_cuda.py
```

# Quadratic Programming Problem


### function `qp1`

Given a quadratic programming problem that aims to minimize the quadratic objective function, subject to a set of linear constraints:

$$
\min_{x} \frac12 x^TPx+q^Tx \\
\text{s.t. } lb \leq x \leq rb, \\
Gx \leq h
$$

where $P$ is a $(n,n)$ positive definite matrix, $q$ is an $n$-dimensional vector, $G$ is a $(g,n)$ matrix, $h$ is a $g$-dimensional vector, and $lb,rb$ are $n$-dimensional vectors defining the lower and upper bounds for variable $x$.

#### Input Parameters

- **P** : A numpy.ndarray or similar buffer type with shape $(n,n)$, for the quadratic coefficient matrix.
- **q** (optional) : A numpy.ndarray or similar buffer type with shape $n$, for the linear coefficient matrix.
- **G** (optional) : A numpy.ndarray or similar buffer type with shape $(n,n)$, for the linear inequality constraint coefficient matrix.
- **h** (optional) : A numpy.ndarray or similar buffer type with shape $n$, for the right-hand side vector for linear inequalities.
- **lb** (optional) : A numpy.ndarray or similar buffer type with shape $n$, for the lower bounds of variables.
- **rb** (optional) : A numpy.ndarray or similar buffer type with shape $n$, for the upper bounds of variables.

#### Output Parameters

- **x**: A numpy.ndarray wieh shape $n$, for the solution to the problem

```python
# sample data
import numpy as np
X = np.random.normal(size=(n_sample, n_variable))
P = X.T @ X + 1e-9 * np.eye(n_variable)  # add 1e-9 to diagonals to make it strict positive definite
q = np.random.normal(size=n_sample)
lb = -np.ones(n_variable)
rb = np.ones(n_variable)
# cpu version
import qptools
x = qptools.qp1(P=P, q=q, lb=lb, rb=rb)
# cuda version
import qptools.cuda
handle = cuda.Handle()
x = qptools.cuda.qp1(handle, P=P, q=q, lb=lb, rb=rb)
```

### function `qp2`

$$
\min_{x} \frac12 x^TPx+q^Tx \\
\text{s.t. } lb \leq \frac{x}{\sum(x)} \leq rb, \\
Gx \leq h
$$

where $P$ is a $(n,n)$ positive definite matrix, $q$ is an $n$-dimensional vector, $G$ is a $(g,n)$ matrix, $h$ is a $g$-dimensional vector, and $lb,rb$ are $n$-dimensional vectors defining the lower and upper bounds for standardized variable $\frac{x}{\sum(x)}$.

#### Input Parameters

- **P** : A numpy.ndarray or similar buffer type with shape $(n,n)$, for the quadratic coefficient matrix.
- **q** (optional) : A numpy.ndarray or similar buffer type with shape $n$, for the linear coefficient matrix.
- **G** (optional) : A numpy.ndarray or similar buffer type with shape $(n,n)$, for the linear inequality constraint coefficient matrix.
- **h** (optional) : A numpy.ndarray or similar buffer type with shape $n$, for the right-hand side vector for linear inequalities.
- **lb** (optional) : A numpy.ndarray or similar buffer type with shape $n$, for the lower bounds of variables.
- **rb** (optional) : A numpy.ndarray or similar buffer type with shape $n$, for the upper bounds of variables.

#### Output Parameters

- **x**: A numpy.ndarray wieh shape $n$, for the solution to the problem


```python
# sample data
import numpy as np
X = np.random.normal(size=(n_sample, n_variable))
P = X.T @ X + 1e-9 * np.eye(n_variable)  # add 1e-9 to diagonals to make it strict positive definite
q = np.random.normal(size=n_sample)
lb = -np.ones(n_variable)
rb = np.ones(n_variable)
# cpu version
import qptools
x = qptools.qp2(P=P, q=q, lb=lb, rb=rb)
# cuda version
import qptools.cuda
handle = cuda.Handle()
x = qptools.cuda.qp2(handle, P=P, q=q, lb=lb, rb=rb)
```
# References
