#pragma once
__global__ void cuda_fill(int len, double* a, double val);
__global__ void cuda_copy(int len, double* a, const double* vals);
__global__ void cuda_copy(int len, double* a, const double* vals, double alpha);
__global__ void cuda_fmcopy(int len, double* a, const double* vals1, const double* vals2);
__global__ void cuda_add(int len, double* a, const double* vals, double alpha);
__global__ void cuda_fmadd(int len, double* a, const double* vals1, const double* vals2);

class cumatrix{
    public:
        const int nrows, ncols, size;
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
        void copy(const double* vals, double alpha);
        void fmcopy(const double* vals1, const double* vals2);
        void add(double val);
        void add(const double* vals, double alpha);
        void fmadd(const double* vals1, const double* vals2);
        void scal(double val);
        void scal(const double* vals);
        void divide(const double* vals);
        double min() const;
        double max() const;
        double sum() const;
        double nrm2() const;
        double dot(const double* vals) const;
};