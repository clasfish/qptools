#include <iostream>
#include <algorithm>
#include <numeric>
#include "matrix_base.h"

matrix::matrix(int nrows, int ncols):
    nrows(nrows), ncols(ncols), size(nrows*ncols), begin((size>0)?(new double[size]):nullptr){}

matrix::matrix(int nrows, int ncols, double val):
    matrix(nrows, ncols){std::fill(begin, begin+size, val);}

matrix::matrix(const matrix& other):
    matrix(other.nrows, other.ncols){std::copy(other.begin, other.begin+size, begin);}

matrix::~matrix(){
    delete[] begin;
}

void matrix::display() const{
    int i, j;
    double *iter;
    for(i=0; i<nrows; i++){
        iter = begin+i;
        for(j=0; j<ncols-1; j++, iter+=nrows) std::cout << *iter << " ";
        std::cout << *iter << std::endl;
    }
}

void matrix::_display(int len) const{
    int i;
    double *iter=begin;
    for(i=0; i<len-1; i++, iter++) std::cout << *iter << " ";
    std::cout << *iter << std::endl;
}

void matrix::fill(double val){
    std::fill(begin, begin+size, val);
}

void matrix::fill_iota(){
    std::iota(begin, begin+size, 0.0);
}

void matrix::copy(const double* vals){
    std::copy(vals, vals+size, begin);
}

void matrix::add(double val){
    double *iter=begin, *end=begin+size;
    for(; iter<end; iter++) *iter += val;
}

void matrix::add(const double* vals, double alpha){
    double *iter=begin, *end=begin+size;
    if(alpha==1.0) for(;iter<end; iter++, vals++) *iter += *vals;
    else if(alpha==-1.0) for(;iter<end; iter++, vals++) *iter -= *vals;
    else for(;iter<end; iter++, vals++) *iter += alpha * *vals;
}

void matrix::scal(double val){
    double *iter=begin, *end=begin+size;
    for(; iter<end; iter++) *iter *= val;
}

void matrix::scal(const double* vals){
    double *iter=begin, *end=begin+size;
    for(; iter<end; iter++, vals++) *iter *= *vals;
}

void matrix::divide(const double* vals){
    double *iter=begin, *end=begin+size;
    for(; iter<end; iter++, vals++) *iter /= *vals;
}

void matrix::sort(){
    std::sort(begin, begin+size);
}

void matrix::reverse(){
    std::reverse(begin, begin+size);
}

double matrix::min() const{
    return *std::min_element(begin, begin+size);
}

double matrix::max() const{
    return *std::max_element(begin, begin+size);
}

double matrix::sum() const{
    return std::accumulate(begin, begin+size, 0.0);
}


