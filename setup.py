import os
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
os.environ

ext_modules = []
cmdclass = dict()

if True:
    core = Pybind11Extension(
        "qptools.core",
        libraries = ['mkl_rt'],
        sources = [
            "c/core/src/matrix_base.cpp", 
            "c/core/src/qp1.cpp",
            "c/core/src/qp2.cpp",
            "c/core/src/core_bind.cpp"
            ],
        include_dirs = ["c/core/include"],
        extra_compile_args = ['-O3', '-Wall', '-Wextra', '-finline']
    )
    ext_modules.append(core)

if False:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    if not torch.cuda.is_available:
        print("CUDA is not available, please install the cpu version")
        exit(1)
    cudacore = CUDAExtension(
        "qptools.cudacore",
        sources = [
            "c/cudacore/src/cumatrix_base.cu",
            "c/cudacore/src/cudacore_bind.cu"
        ],
        include_dirs = ["c/cudacore/include"],
        extra_compile_args = {
            'cxx': ['-O3', '-Wall', '-Wextra'],
            'nvcc': ['-O3']
        },
    )
    ext_modules.append(cudacore)
    cmdclass['build_ext'] = BuildExtension


setup(
    name = "qptools",
    version = "1.0.1", 
    description = "quadratic optimization tools",
    ext_modules = ext_modules,
    packages = ["qptools"],
    package_dir = {"qptools": "python"},
    url = "https://gitee.com/clawfish/qptools",
    author = "clawfish",
    python_requires = ">=3.8",
    classifiers = [
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Education"
    ],
    cmdclass = cmdclass
)