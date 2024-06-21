#include "matrix_util.cuh"
#include "matrix_base.cuh"
#include "matrix_read.cuh"
#include "qp.cuh"


int main(){
    matrix *P = read_csv("../../../../data/P.csv", ',');
    matrix *lb = read_csv("../../../../data/lb.csv", ',');
    matrix *rb = read_csv("../../../../data/rb.csv", ',');
    matrix *G0 = read_csv("../../../../data/G0.csv", ',');
    matrix *h0 = read_csv("../../../../data/h0.csv", ',');
    matrix *G1 = read_csv("../../../../data/G1.csv", ',');
    matrix *h1 = read_csv("../../../../data/h1.csv", ',');
    matrix *G2 = read_csv("../../../../data/G2.csv", ',');
    matrix *h2 = read_csv("../../../../data/h2.csv", ',');
    CublasHandle cublas_handle = CublasHandle();
    CusolverHandle cusolver_handle = CusolverHandle();
    // qp1 - 1
    qp1 solver1(cublas_handle, cusolver_handle, P, nullptr, lb, rb, G0, h0);
    matrix *x1 = solver1.solve();
    x1->_display(5);
    // qp1 - 2
    qp1 solver2(cublas_handle, cusolver_handle, P, nullptr, nullptr, nullptr, G1, h1);
    matrix *x2 = solver2.solve();
    x2->_display(5); 
    // qp2 - 3
    qp2 solver3(cublas_handle, cusolver_handle, P, nullptr, lb, rb, G0, h0);
    matrix *x3 = solver3.solve();
    x3->_display(5);
    // qp2 - 4
    qp2 solver4(cublas_handle, cusolver_handle, P, nullptr, nullptr, nullptr, G2, h2);
    matrix *x4 = solver4.solve();
    x4->_display(5);
}

