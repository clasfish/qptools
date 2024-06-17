# Dependencies

### Dependencies for CPU version:

- setuptools
- pybind11
- numpy
### Dependencies for GPU version:

- setuptools
- pybind11
- pip >= 23.0
- numpy
- pytorch(cuda)

# Installation

*Notes: developing*
<!-- 
CPU version:

    pip install qptools

GPU version:

    pip install qptools --enable-gpu
-->
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

#### Example

```python
import numpy as np
import qptools
P = np.array([[5.0, 3.0, 1.0], [3.0, 6.0, 2.0], [1.0, 2.0, 7.0]])
lb = np.array([-2.0, 1.0, 3.0])
rb = np.array([3.0, 5.0, 4.0])
x = qptools.qp1(P=P, lb=lb, rb=rb)
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

# References
