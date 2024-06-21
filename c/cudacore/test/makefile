CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++14 -O3
NVCCFLAGS = -std=c++14 -O3
PYTHON_INCLUDE = $(python -c "import sysconfig; print(sysconfig.get_path('include'))")
IPATH = -I../include
LPATH = -lcublas -lcusolver

SRC = matrix_base.o matrix_read.o matrix_util.o qp.o test.o

test: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(LPATH) $^ -o $@

%.o: ../src/%.cu
	$(NVCC) -g -c $(NVCCFLAGS) $(IPATH) $^ -o $@

%.o: %.cu
	$(NVCC) -g -c $(NVCCFLAGS) $(IPATH) $^ -o $@

util.o: util.cpp
	$(CXX) -g -c $(CXXFLAGS) $(IPATH) $^ -o $@

valgrind: test
	valgrind --leak-check=full ./test
	
check:
	@echo $(PYTHON_INCLUDE)

clean:
	rm -f *.o
	rm -f test