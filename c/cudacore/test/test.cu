#include "util.h"
#include "cumatrix_util.h"
#include "cumatrix_base.h"
#include "cumatrix_read.h"
#include "qp.h"
#include <cstring>

#include <cmath>
#include <thrust/reduce.h>


int main(){
    cumatrix *P = read_csv("../../../../data/P.csv", ',');
    cumatrix *lb = read_csv("../../../../data/lb.csv", ',');
    cumatrix *rb = read_csv("../../../../data/rb.csv", ',');
    cumatrix *G0 = read_csv("../../../../data/G0.csv", ',');
    cumatrix *h0 = read_csv("../../../../data/h0.csv", ',');
    cumatrix *G1 = read_csv("../../../../data/G1.csv", ',');
    cumatrix *h1 = read_csv("../../../../data/h1.csv", ',');
    cumatrix *G2 = read_csv("../../../../data/G2.csv", ',');
    cumatrix *h2 = read_csv("../../../../data/h2.csv", ',');

    cublasHandle_t cublas_handle = create_cublas_handle();
    cusolverDnHandle_t cusolver_handle = create_cusolver_handle();

    qp1 solver1(cublas_handle, cusolver_handle, P, nullptr, lb, rb, G0, h0);
    cumatrix *x = solver1.solve();
    x->_display(5);
    qp1 solver2(cublas_handle, cusolver_handle, P, nullptr, nullptr, nullptr, G1, h1);
    x = solver2.solve();
    x->_display(5);
    cublasDestroy(cublas_handle);
    cusolverDnDestroy(cusolver_handle);

}

