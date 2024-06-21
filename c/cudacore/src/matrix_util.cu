#include "matrix_util.cuh"


CublasHandle::CublasHandle(){
    cublasCreate(&handle);
}

CublasHandle::~CublasHandle(){
    cublasDestroy(handle);
}

CusolverHandle::CusolverHandle(){
    cusolverDnCreate(&handle);
}

CusolverHandle::~CusolverHandle(){
    cusolverDnDestroy(handle);
}
