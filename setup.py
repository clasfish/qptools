import os
import sys
import subprocess
import re
import copy
from configparser import ConfigParser
from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext

# config
cwd = os.path.dirname(os.path.realpath(__file__))
_config = ConfigParser()
_config.read(os.path.join(cwd, "setup.cfg"))
def get_list_arg(section, option):
    return _config.get(section, option).split(":")

def get_dict_arg(section, option):
    return dict((item.split(':')) for item in _config.get(section, option).split(','))

COMMON_MSVC_FLAGS = get_list_arg("flag", "COMMON_MSVC_FLAGS")
MSVC_IGNORE_CUDAFE_WARNINGS = get_list_arg("flag", "MSVC_IGNORE_CUDAFE_WARNINGS")
COMMON_NVCC_FLAGS = get_list_arg("flag", "COMMON_NVCC_FLAGS")
PLAT_TO_VCVARS = get_dict_arg("dict", "PLAT_TO_VCVARS")
CXX_FLAGS = get_list_arg("flag", "CXX_FLAGS")
NVCC_FLAGS = get_list_arg("flag", "NVCC_FLAGS")

# environ
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')
SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()
# cxx
def get_cxx_compiler():
    if IS_WINDOWS:
        compiler = os.environ.get("CXX", "cl")
    else:
        compiler = os.environ.get("CXX", "c++")
    return compiler

# libs
libs = os.listdir(os.path.join(os.environ["CONDA_PREFIX"], "lib"))
if any("libmkl" in lib for lib in libs):
    libraries = ["mkl_rt"]
elif any("libopenblas" in lib for lib in libs):
    libraries = ["blas", "lapack"]
else:
    raise ValueError("The installation needs mkl or openblas")

# cuda
def find_cuda_home():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CODA_PATH")
    if cuda_home is None:
        try:
            which = 'where' if IS_WINDOWS else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'], stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            if IS_WINDOWS:
                cuda_homes = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    return cuda_home

USE_CUDA = int(os.getenv('USE_CUDA', 0))
CUDA_HOME = find_cuda_home()
CUDA_LD_FLAGS = []
if CUDA_HOME is not None:
    NVCC = os.path.join(CUDA_HOME, "bin", "nvcc")
    CUDA_LD_FLAGS = [f'-L{os.path.join(CUDA_HOME, subdir)}' for subdir in ['lib', 'lib64']]

if USE_CUDA and not CUDA_HOME:
    raise RuntimeError("CUDA is not available, please check the CUDA_HOME or CUDA_PATH")

def append_std17_if_no_std_present(compiler_type, cflags):
    cpp_flag_prefix = '/std:' if compiler_type == 'msvc' else '-std='
    cpp_flag = cpp_flag_prefix + 'c++17'
    if not any(cflag.startswith(cpp_flag_prefix) for cflag in cflags):
        cflags.append(cpp_flag)

def unix_cuda_flags(cflags):
    cflags = COMMON_NVCC_FLAGS + ['--compiler-options', "'-fPIC'"] + cflags
    _ccbin = os.getenv("CC")
    if _ccbin is not None and not any(cflag.startswith(('--ccbin', '--compiler-binder')) for cflag in cflags):
        cflags.extend(['--ccbin', _ccbin])
    return cflags

class BuildExtension(build_ext):
    @classmethod
    def with_options(cls, **options):
        class cls_with_options(cls):
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)
        return cls_with_options
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_extensions(self):
        compiler_name = self.compiler.compiler_cxx[0] if hasattr(self.compiler, "compiler_cxx") else get_cxx_compiler()
        cuda_ext = False
        extension_iter = iter(self.extensions)
        for extension in self.extensions:
            for source in extension.sources:
                if os.path.splitext(source)[1] in ['.cu', '.cuh']:
                    cuda_ext = True
                    break
            if cuda_ext:
                break
        #if cuda_ext:
        #    check_cuda_version(compiler_name, compiler_version)
        for extension in self.extensions:
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []
            # define torch_extension_name
            # add_gnu_cpp_abi_flag
        self.compiler.src_extensions += ['.cu', '.cuh']
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile
        # -------------------- wrap_compile
        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if os.path.splitext(src)[1] in ['.cu', '.cuh']:
                    self.compiler.set_executable('compiler_so', NVCC)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    cflags = unix_cuda_flags(cflags)
                else:
                    if isinstance(cflags, dict):
                        cflags = cflags['cxx']
                append_std17_if_no_std_present(self.compiler.compiler_type, cflags)
                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                self.compiler.set_executable('compiler_so', original_compiler)
        def win_wrap_single_compile(*args, **kwargs):
            pass
        # -------------------- compile
        if self.compiler.compiler_type == 'msvc':
            self.compiler.compile = win_wrap_single_compile
        else:
            self.compiler._compile = unix_wrap_single_compile
        build_ext.build_extensions(self)

ext_modules = []
cmdclass = dict()

# core module
core = Pybind11Extension(
    "qptools.core",
    libraries = libraries,
    sources = [
        "c/core/src/matrix_base.cpp", 
        "c/core/src/qp.cpp",
        "c/core/src/core_bind.cpp"
        ],
    include_dirs = ["c/core/include"],
    extra_compile_args = CXX_FLAGS
)
ext_modules.append(core)

# cuda core module
if USE_CUDA:
    cudacore = Pybind11Extension(
        "qptools.cudacore",
        sources = [
            "c/cudacore/src/matrix_base.cu",
            "c/cudacore/src/matrix_util.cu",
            "c/cudacore/src/qp.cu",
            "c/cudacore/src/cudacore_bind.cu"
        ],
        include_dirs = ["c/cudacore/include"],
        extra_compile_args = {
            'cxx': CXX_FLAGS,
            'nvcc': NVCC_FLAGS
        },
        extra_link_args = CUDA_LD_FLAGS + ['-Wl,--no-as-needed', '-lcudart', '-lcublas', '-lcusolver']
    )
    ext_modules.append(cudacore)
    cmdclass = {'build_ext': BuildExtension}


setup(
    name = "qptools",
    version = "1.0.2", 
    description = "Fast quadratic programming tools (with CUDA support)",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
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
