#pragma once
//#include "cuda_runtime.h"

class cumatrix{
    public:
        const int nrows, ncols, size;
        //const dim3 blockSize;
        //const dim3 gridSize;
        const int blockSize;
        const int gridSize;
        double *begin;
        cumatrix(int nrows, int ncols);
        cumatrix(int nrows, int ncols, double val);
        ~cumatrix();
        void display() const;
        void _display(int len) const;
        void fill(double val);
        void copy(const double* vals);
        void add(double val);
        void add(const double* vals, double alpha);
        void fmadd(const double* vals1, const double* vals2);
        void scal(double val);
        void scal(const double* vals);
        void divide(const double* vals);
};