#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <string>
#include <cuda_runtime.h>
#include "cumatrix_base.h"
#include "util.h"

void get_file_dimensions(const std::string& path, char sep, int& nrows, int& ncols){
    std::ifstream file(path);
    if(!file.is_open()) throw std::runtime_error("Cannot open file:" + path);
    std::string line, item;
    nrows = 1;
    ncols = 0;
    std::getline(file, line);
    std::istringstream lineStream(line);
    while(std::getline(lineStream, item, sep)) ncols++;
    while(std::getline(file, line)) nrows++;
    file.close();
}

cumatrix* read_csv(const std::string& path, char sep){
    int i, j, nrows, ncols;
    get_file_dimensions(path, sep, nrows, ncols);
    std::string line, item;
    std::ifstream file(path);
    if(!file.is_open()) throw std::runtime_error("Cannot open file:" + path);
    double *a = new double[nrows*ncols], *iter;
    cumatrix *A = new cumatrix(nrows, ncols);
    for(i=0; i<nrows; i++){
        iter = a + i;
        std::getline(file, line);
        std::istringstream lineStream(line);
        for(j=0; j<ncols; j++){
            std::getline(lineStream, item, sep);
            *iter = stof(item);
            iter += nrows;
        }
    }
    file.close();
    cudaMemcpy(A->begin, a, nrows*ncols*sizeof(double), cudaMemcpyHostToDevice);
    free(a);
    return A;
}