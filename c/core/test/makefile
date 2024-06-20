SRC = matrix_base.o matrix_read.o util.o qp1.o qp2.o test.o
CXX ?= g++
CXXFLAGS += -std=c++14 -O3 -finline -Wall -Wextra
IPATH += -I../include
LPATH += -L$(CONDA_PREFIX)/lib
LIBMKL ?= $(wildcard $(CONDA_PREFIX)/lib/libmkl*)
LIBOPENBLAS ?= $(wildcard $(CONDA_PREFIX)/lib/libopenblas*)

ifeq ($(LIBMKL),)
ifeq ($(LIBOPENBLAS),)
else
	LDFLAGS += -lblas -llapack
endif
else
	LDFLAGS += -lmkl_rt
endif


test: $(SRC)
	$(CXX) $(CXXFLAGS) $(LPATH) $(LDFLAGS) $^ -o $@

%.o: ../src/%.cpp
	$(CXX) -g -c $(CXXFLAGS) $(IPATH) $^ -o $@

%.o: %.cpp
	$(CXX) -g -c $(CXXFLAGS) $(IPATH) $^ -o $@


debug:
	@echo $(LIBMKL)
	@echo $(LIBOPENBLAS)
	@echo $(LDFLAGS)
	
clean:
	rm *.o