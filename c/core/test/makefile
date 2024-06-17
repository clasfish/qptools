CXX = g++
CXXFLAGS = -std=c++14 -O3 -finline -Wall -Wextra
PYTHON_INCLUDE = $(python -c "import sysconfig; print(sysconfig.get_path('include'))")
IPATH = -I../include
LPATH = -L$(CONDA_PREFIX)/lib -lmkl_rt

SRC = matrix_base.o matrix_read.o util.o qp1.o qp2.o util.o test.o

test: $(SRC)
	$(CXX) $(CXXFLAGS) $(LPATH) $^ -o $@

%.o: ../src/%.cpp
	$(CXX) -g -c $(CXXFLAGS) $(IPATH) $^ -o $@

%.o: %.cpp
	$(CXX) -g -c $(CXXFLAGS) $(IPATH) $^ -o $@

check:
	@echo $(PYTHON_INCLUDE)

clean:
	rm *.o