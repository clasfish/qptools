CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++14 -O3 -finline -Wall -Wextra
NVCCFLAGS = -std=c++14 -O3
PYBIND_INCLUDE = $(shell python3 -m pybind11 --includes)
PYTHON_SUFFIX = $(shell python3-config --extension-suffix)
IPATH = -Ic/cudacore/include $(PYBIND_INCLUDE)
LPATH = -lcublas -lcusolver
# project
TARGET = cudacore$(PYTHON_SUFFIX)
SOURCES = build/cumatrix_base.o build/cumatrix_util.o build/qp1.o build/cudacore_bind.o

all: $(TARGET)
	@echo completed

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -shared $(LPATH) $^ -o $@
	
build/%.o: c/cudacore/src/%.cu
	$(NVCC) $(NVCCFLAGS) -c -Xcompiler "-fPIC" $(IPATH)  $^ -o $@

check:
	echo $(PYBIND_INCLUDE)
	echo $(PYTHON_SUFFIX)
	
clean:
	# build
	rm -rf qptools.egg-info
	rm -rf dist
	rm -f build/*
	# test
	rm -f test/*.o
	rm -f c/core/test/*.o c/core/test/test
	rm -f c/cudacore/test/*.o c/cuadcore/test/test
	# so
	rm *.so