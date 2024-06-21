#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include "matrix_base.cuh"

__global__ void cuda_fill(int len, double* a, double val){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] = val;
}

__global__ void cuda_copy(int len, double* a, const double* vals){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] = vals[i];
}

__global__ void cuda_copy(int len, double* a, const double* vals, double alpha){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] = vals[i] * alpha;
}

__global__ void cuda_fmcopy(int len, double* a, const double* vals1, const double* vals2){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) a[i] = vals1[i] * vals2[i];
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


matrix::matrix(int nrows, int ncols):
    nrows(nrows),
    ncols(ncols),
    size(nrows * ncols),
    blockSize((size<=1024)?size:1024),
    gridSize((size<=1024)?1:(size+1023)/1024)
{
    if(size < 0) throw std::runtime_error("The size of a matrix should non-negative"); 
    if(size == 0) begin = nullptr;
    else{
        cudaError_t status = cudaMalloc((void**)&begin, size*sizeof(double));
        if(status != cudaSuccess) throw std::runtime_error("Falied to allocate GPU memory with cudaMalloc: " + std::string(cudaGetErrorString(status)));
    }
}



matrix::matrix(int nrows, int ncols, double val):
    matrix(nrows, ncols){
    cuda_fill<<<blockSize, gridSize>>>(size, begin, val);
}

matrix::~matrix(){
    cudaFree(begin);
}

void matrix::display() const{
    if(size <= 0) return;
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

void matrix::_display(int len) const{
    if((size <= 0) || (len <= 0)) return;
    if(len > size) throw std::runtime_error("The length should not be larger than the size of the matrix");
    int i;
    double *a = new double[size], *iter=a;
    cudaMemcpy(a, begin, size * sizeof(double), cudaMemcpyDeviceToHost);
    for(i=0; i<len-1; i++, iter++) std::cout << *iter << ",";
    std::cout << *iter << std::endl;
    free(a);
}

void matrix::fill(double val){
    cuda_fill<<<blockSize, gridSize>>>(size, begin, val);
}

void matrix::copy(const double* vals){
    cuda_copy<<<blockSize, gridSize>>>(size, begin, vals);
}

void matrix::copy(const double* vals, double alpha){
    cuda_copy<<<blockSize, gridSize>>>(size, begin, vals, alpha);
}

void matrix::fmcopy(const double* vals1, const double* vals2){
    cuda_fmcopy<<<blockSize, gridSize>>>(size, begin, vals1, vals2);
}

void matrix::add(double val){
    cuda_add<<<blockSize, gridSize>>>(size, begin, val);
}

void matrix::add(const double* vals, double alpha){
    cuda_add<<<blockSize, gridSize>>>(size, begin, vals, alpha);
}

void matrix::fmadd(const double* vals1, const double* vals2){
    cuda_fmadd<<<blockSize, gridSize>>>(size, begin, vals1, vals2);
}

void matrix::scal(double val){
    cuda_scal<<<blockSize, gridSize>>>(size, begin, val);
}

void matrix::scal(const double* vals){
    cuda_scal<<<blockSize, gridSize>>>(size, begin, vals);
}

void matrix::divide(const double* vals){
    cuda_divide<<<blockSize, gridSize>>>(size, begin, vals);
}


struct op_square{
    __device__ inline double operator()(double x) const{
        return x * x;
    }
};

struct op_plus{
    __device__ inline double operator()(double x, double y) const{
        return x + y;
    }
};

struct op_min{
    __device__ inline double operator()(double x, double y) const{
        return (x>y)?y:x;
    }
};

struct op_max{
    __device__ inline double operator()(double x, double y) const{
        return (x>y)?x:y;
    }
};



double matrix::min() const{
    double hres;
    double* dres = thrust::min_element(thrust::device, begin, begin+size);
    cudaMemcpy(&hres, dres, sizeof(double), cudaMemcpyDeviceToHost);
    return hres;
}

double matrix::max() const{
    double hres;
    double* dres = thrust::max_element(thrust::device, begin, begin+size);
    cudaMemcpy(&hres, dres, sizeof(double), cudaMemcpyDeviceToHost);
    return hres;
}

double matrix::sum() const{
    return thrust::reduce(thrust::device, begin, begin+size);
}

double matrix::nrm2() const{
    return std::sqrt(thrust::transform_reduce(thrust::device, begin, begin+size, op_square(), 0.0, op_plus()));
}

double matrix::dot(const double* vals) const{
    return thrust::inner_product(thrust::device, begin, begin+size, vals, 0.0);
}
