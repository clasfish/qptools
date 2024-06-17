#include "cumatrix_base.h"
#include "cumatrix_util.h"

int main(){
    matrix *A = new matrix(3, 4, 1.0);
    cumatrix *cuA = matrix_to_cumatrix(A);
    display(cuA, 5); 
}