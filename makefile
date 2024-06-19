NVCC = nvcc
NVCCFLAGS = -std=c++14 -O3
IPATH = -Ic/cudacore/include $(shell python3 -m pybind11 --includes)
LPATH = -lcublas -lcusolver
SOURCES = build/cumatrix_base.o build/cumatrix_util.o build/cuqp1.o build/cudacore_bind.o
TARGET ?= cudacore$(shell python3-config --extension-suffix) 
PREFIX ?= .

all: $(TARGET)
	cp $(TARGET) $(PREFIX)/
	@echo "make successed!"

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -shared $(LPATH) $^ -o $@
	
build/%.o: c/cudacore/src/%.cu
	$(NVCC) $(NVCCFLAGS) -c -Xcompiler "-fPIC" $(IPATH)  $^ -o $@

check:
	echo $(PREFIX)
	echo $(TARGET)

ls:
	@ls /root/.local/conda/envs/pytest/lib/python3.12/site-packages/qptools
	@ls /root/.local/conda/envs/pytest/lib/python3.12/site-packages/cudacore.cpython-312-x86_64-linux-gnu.so
clean:
	# build
	rm -rf qptools.egg-info
	rm -rf dist
	rm -rf build/*
	rm -rf python/__pycache__
	# test
	rm -f test/*.o
	rm -f c/core/test/*.o c/core/test/test
	rm -f c/cudacore/test/*.o c/cuadcore/test/test
	# so
