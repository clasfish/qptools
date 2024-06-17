#include <iostream>
#include <iomanip>
#include "cuda_runtime.h"
#include "cumatrix_base.h"

__global__ void cuda_fill(int len, double* a, double val){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] = val;
}

__global__ void cuda_copy(int len, double* a, const double* vals){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] = vals[i];
}

__global__ void cuda_add(int len, double* a, double val){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] += val;
}

__global__ void cuda_add(int len, double* a, const double* vals, double alpha){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] += alpha * vals[i];
}

__global__ void cuda_fmadd(int len, double* a, const double* vals1, const double* vals2){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] += vals1[i] * vals2[i];
}

__global__ void cuda_scal(int len, double* a, double val){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] *= val;
}

__global__ void cuda_scal(int len, double* a, const double* vals){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] *= vals[i];
}

__global__ void cuda_divide(int len, double* a, const double* vals){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] /= vals[i];
}


cumatrix::cumatrix(int nrows, int ncols):
    nrows(nrows),
    ncols(ncols),
    size(nrows * ncols),
    blockSize((size>=1024)?1024:size), //dim3(blockSize)
    gridSize((size+blockSize-1)/blockSize) //dim3(gridSize)
{
    cudaMalloc((void**)&begin, size*sizeof(double));
}



cumatrix::cumatrix(int nrows, int ncols, double val):
    cumatrix(nrows, ncols){
    cuda_fill<<<blockSize, gridSize>>>(size, begin, val);
}

cumatrix::~cumatrix(){
    cudaFree(begin);
}

void cumatrix::display() const{
    int i, j;
    double *a = new double[size], *iter=a;
    cudaMemcpy(a, begin, size * sizeof(double), cudaMemcpyDeviceToHost);
    for(i=0; i<nrows; i++){
        iter = a+i;
        for(j=0; j<ncols-1; j++, iter+=nrows) std::cout << *iter << " ";
        std::cout << *iter << std::endl;
    }
    free(a);
}

void cumatrix::_display(int len) const{
    int i;
    double *a = new double[size], *iter=a;
    cudaMemcpy(a, begin, size * sizeof(double), cudaMemcpyDeviceToHost);
    for(i=0; i<len-1; i++, iter++) std::cout << std::setprecision(9) << *iter << " ";
    std::cout << std::setprecision(9) << *iter << std::endl;
    free(a);
}

void cumatrix::fill(double val){
    cuda_fill<<<blockSize, gridSize>>>(size, begin, val);
}

void cumatrix::copy(const double* vals){
    cuda_copy<<<blockSize, gridSize>>>(size, begin, vals);
}

void cumatrix::add(double val){
    cuda_add<<<blockSize, gridSize>>>(size, begin, val);
}

void cumatrix::add(const double* vals, double alpha){
    cuda_add<<<blockSize, gridSize>>>(size, begin, vals, alpha);
}

void cumatrix::fmadd(const double* vals1, const double* vals2){
    cuda_fmadd<<<blockSize, gridSize>>>(size, begin, vals1, vals2);
}

void cumatrix::scal(double val){
    cuda_scal<<<blockSize, gridSize>>>(size, begin, val);
}

void cumatrix::scal(const double* vals){
    cuda_scal<<<blockSize, gridSize>>>(size, begin, vals);
}

void cumatrix::divide(const double* vals){
    cuda_divide<<<blockSize, gridSize>>>(size, begin, vals);
}

