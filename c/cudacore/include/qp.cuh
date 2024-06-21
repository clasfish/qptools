#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#define MIN(i, j) (((i) < (j)) ? (i) : (j))
#define MAX(i, j) (((i) > (j)) ? (i) : (j))

class qp1{
    public:
        qp1(const CublasHandle& _cublas_handle, const CusolverHandle& _cusolver_handle, const matrix* P, const matrix* q, const matrix* lb, const matrix* rb, const matrix* G, const matrix* h);
        ~qp1();
        matrix* solve();
    private:
        cublasHandle_t cublas_handle;
        cusolverDnHandle_t cusolver_handle;
        const int n, lbdim, rbdim, bdim, gdim, cdim;
        const int nblock, ngrid, gblock, ggrid, cblock, cgrid;
        const matrix *P, *q, *lb, *rb, *G, *h;
        matrix *L, *d, *dsq, *Gd;
        int *d_info;
        int d_worklen;
        double *d_work;
        void fG(char trans, const matrix* x, matrix* y) const;
        void kktfactor();
        void kktsolver(matrix* x, matrix* y, matrix* u) const;
};

class qp2{
    public:
        qp2(const CublasHandle& _cublas_handle, const CusolverHandle& _cusolver_handle, const matrix* P, const matrix* q, const matrix* lb, const matrix* rb, const matrix* G, const matrix* h);
        ~qp2();
        matrix* solve();
    private:
        cublasHandle_t cublas_handle;
        cusolverDnHandle_t cusolver_handle;
        const int n, lbdim, rbdim, bdim, gdim, cdim;
        const int nblock, ngrid, bblock, bgrid, gblock, ggrid, cblock, cgrid;
        const matrix *P, *q, *lb, *rb, *G, *h;
        matrix *L, *bd, *d, *dsq, *Gd;
        int *d_info;
        int d_worklen;
        double *d_work;
        void fG(char trans, const matrix* x, matrix* y) const;
        void kktfactor();
        void kktsolver(matrix* x, matrix* y, matrix* u) const;
};