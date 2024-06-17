#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include "matrix_base.h"


void get_file_dimensions(int& nrows, int& ncols, const std::string& path, char sep){
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

matrix* read_csv(const std::string& path, char sep){
    int i, j, nrows, ncols;
    get_file_dimensions(nrows, ncols, path, sep);
    std::string line, item;
    std::ifstream file(path);
    if(!file.is_open()) throw std::runtime_error("Cannot open file:" + path);
    double *iter;
    matrix *a = new matrix(nrows, ncols);
    for(i=0; i<nrows; i++){
        iter = a->begin + i;
        std::getline(file, line);
        std::istringstream lineStream(line);
        for(j=0; j<ncols; j++){
            std::getline(lineStream, item, sep);
            *iter = stof(item);
            iter += nrows;
        }
    }
    file.close();
    return a;
}

matrix* random_matrix(int nrows, int ncols, int seed){
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    matrix *a = new matrix(nrows, ncols);
    double *iter=a->begin, *end=a->begin+a->size;
    for(; iter<end; iter++) *iter = distribution(generator);
    return a;
}
