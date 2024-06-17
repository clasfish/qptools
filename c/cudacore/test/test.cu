#include "util.h"
#include "matrix_read.h"
#include "cumatrix_base.h"
#include <cstring>

cumatrix *cumatrix_read(const std::string& path, char sep){
    matrix *a = read_csv(path, sep);
    cumatrix *A = new cumatrix(a->nrows, a->ncols);
    cudaMemcpy(A->begin, a->begin, A->size * sizeof(double), cudaMemcpyHostToDevice);
}

int main(){
    cumatrix *P = cumatrix_read("../../../../data/P.csv", ',');
    cumatrix *lb = cumatrix_read("../../../../data/lb.csv", ',');
    cumatrix *rb = cumatrix_read("../../../../data/rb.csv", ',');
    cumatrix *G0 = cumatrix_read("../../../../data/G0.csv", ',');
    cumatrix *h0 = cumatrix_read("../../../../data/h0.csv", ',');
    cumatrix *G1 = cumatrix_read("../../../../data/G1.csv", ',');
    cumatrix *h1 = cumatrix_read("../../../../data/h1.csv", ',');
    cumatrix *G2 = cumatrix_read("../../../../data/G2.csv", ',');
    cumatrix *h2 = cumatrix_read("../../../../data/h2.csv", ',');
    P->_display(5);
    qp1 solver1(P, nullptr, lb, rb, G0, h0);
    cumatrix *x = solver1.solve();
}

