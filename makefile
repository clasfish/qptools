CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++14 -O3 -finline -Wall -Wextra
NVCCFLAGS = -std=c++14 -O3
PYBIND_INCLUDE = $(shell python3 -m pybind11 --includes)
PYTHON_SUFFIX = $(shell python3-config --extension-suffix)
IPATH = -Ic/cudacore/include $(PYBIND_INCLUDE)
LPATH = -lcublas
# project
TARGET = cudacore$(PYTHON_SUFFIX)
SOURCES = build/cumatrix_base.o build/cumatrix_util.o build/cuda_bind.o #build/qp1.o

all: $(TARGET)
	@echo completed

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -shared $(LPATH) $^ -o $@
	
test/%.o: cuda/src/%.cu
	$(NVCC) $(NVCCFLAGS) -c -Xcompiler "-fPIC" $(IPATH)  $^ -o $@

test/%.o: cuda/src/%.cpp
	$(CXX) $(CXXFLAGS) -c -fPIC  $(IPATH) $^ -o $@

check:
	echo $(PYBIND_INCLUDE)
	echo $(PYTHON_SUFFIX)
	
clean:
	rm -rf build/*
	rm -rf qptools.egg-info
	rm -rf dist
	rm -f test/*.o
	rm -f main