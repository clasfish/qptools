#pragma once
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cumatrix_base.h"


void check_status(cublasStatus_t status);
cublasHandle_t create_handle();

