#include <iostream>
#include "cumatrix_util.h"


void check_status(cublasStatus_t status){
    if (status != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublas initialized failed:" << status << std::endl;
        exit(1);
    }
}

cublasHandle_t create_handle(){
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    check_status(status);
    return handle;
}

