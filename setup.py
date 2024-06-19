import os
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
ext_modules = []

libs = os.listdir(os.path.join(os.environ["CONDA_PREFIX"], "lib"))
if any("libmkl" in lib for lib in libs):
    libraries = ["mkl_rt"]
elif any("libopenblas" in lib for lib in libs):
    libraries = ["blas", "lapack"]
else:
    raise ValueError("The installation needs mkl or openblas")

core = Pybind11Extension(
    "qptools.core",
    libraries = libraries,
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



'''
projects = ["cumatrix_util", "cumatrix_base", "qp1", "cudacore_bind"]
sources = []
FLAGS = ["-c", "-Xcompiler", "'-fPIC'", "-O3"]
IPATH = [ "-Ic/cudacore/include", f"-I{sysconfig.get_path("include")}", f"-I{pybind11.get_include()}"]
for project in projects:
    src = f"c/cudacore/src/{project}.cu"
    obj = f"build/{project}.o"
    cmd = ["nvcc"] + FLAGS + IPATH + [src, "-o", obj]
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("Command '{}' returned non-zero exit status {}. Output: {}".format(e.cmd, e.returncode, e.stdout))
        print("Error: {}".format(e.stderr))
        exit(1)
    sources.append(obj)
    print(sources)
cudacore = Extension(
    "qptools.cudacore",
    sources = sources,
    library_dirs = ["/usr/local/cuda-11.7/lib64"],
    extra_compile_args=["-O3"],
    extra_link_args = ['-lcudart', '-lcublas', '-lcusolver']
)
ext_modules.append(cudacore)
'''

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
)
