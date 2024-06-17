CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++14 -O3
NVCCFLAGS = -std=c++14 -O3
PYTHON_INCLUDE = $(python -c "import sysconfig; print(sysconfig.get_path('include'))")
IPATH = -I../include -I../../core/include
LPATH = -lcublas

SRC = matrix_base.o matrix_read.o cumatrix_base.o cumatrix_util.o qp1.o util.o test.o

test: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(LPATH) $^ -o $@

%.o: ../src/%.cu
	$(NVCC) -g -c $(NVCCFLAGS) $(IPATH) $^ -o $@

%.o: %.cu
	$(NVCC) -g -c $(NVCCFLAGS) $(IPATH) $^ -o $@

matrix_base.o: ../../core/src/matrix_base.cpp
	$(CXX) -g -c $(CXXFLAGS) $(IPATH) $^ -o $@

matrix_read.o: ../../core/src/matrix_read.cpp
	$(CXX) -g -c $(CXXFLAGS) $(IPATH) $^ -o $@

util.o: util.cpp
	$(CXX) -g -c $(CXXFLAGS) $(IPATH) $^ -o $@

check:
	@echo $(PYTHON_INCLUDE)

clean:
	rm *.o