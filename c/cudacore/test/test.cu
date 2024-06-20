#include "util.h"
#include "cumatrix_util.h"
#include "cumatrix_base.h"
#include "cumatrix_read.h"
#include "cuqp.h"


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
    CublasHandle cublas_handle = CublasHandle();
    CusolverHandle cusolver_handle = CusolverHandle();
    // cuqp1 - 1
    cuqp1 solver1(cublas_handle, cusolver_handle, P, nullptr, lb, rb, G0, h0);
    cumatrix *x1 = solver1.solve();
    x1->_display(5);
    // cuqp1 - 2
    cuqp1 solver2(cublas_handle, cusolver_handle, P, nullptr, nullptr, nullptr, G1, h1);
    cumatrix *x2 = solver2.solve();
    x2->_display(5); 
    // cuqp2 - 3
    cuqp2 solver3(cublas_handle, cusolver_handle, P, nullptr, lb, rb, G0, h0);
    cumatrix *x3 = solver3.solve();
    x3->_display(5);
    // cuqp2 - 4
    cuqp2 solver4(cublas_handle, cusolver_handle, P, nullptr, nullptr, nullptr, G2, h2);
    cumatrix *x4 = solver4.solve();
    x4->_display(5);
}

