#include <iostream>
#include "cumatrix_util.h"


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
