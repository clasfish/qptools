#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#define MIN(i, j) (((i) < (j)) ? (i) : (j))
#define MAX(i, j) (((i) > (j)) ? (i) : (j))

class cuqp1{
    public:
        cuqp1(const CublasHandle& _cublas_handle, const CusolverHandle& _cusolver_handle, const cumatrix* P, const cumatrix* q, const cumatrix* lb, const cumatrix* rb, const cumatrix* G, const cumatrix* h);
        ~cuqp1();
        cumatrix* solve();
    private:
        cublasHandle_t cublas_handle;
        cusolverDnHandle_t cusolver_handle;
        const int n, lbdim, rbdim, bdim, gdim, cdim;
        const int nblock, ngrid, gblock, ggrid, cblock, cgrid;
        const cumatrix *P, *q, *lb, *rb, *G, *h;
        cumatrix *L, *d, *dsq, *Gd;
        int *d_info;
        int d_worklen;
        double *d_work;
        void fG(char trans, const cumatrix* x, cumatrix* y) const;
        void kktfactor();
        void kktsolver(cumatrix* x, cumatrix* y, cumatrix* u) const;
};

class cuqp2{
    public:
        cuqp2(const CublasHandle& _cublas_handle, const CusolverHandle& _cusolver_handle, const cumatrix* P, const cumatrix* q, const cumatrix* lb, const cumatrix* rb, const cumatrix* G, const cumatrix* h);
        ~cuqp2();
        cumatrix* solve();
    private:
        cublasHandle_t cublas_handle;
        cusolverDnHandle_t cusolver_handle;
        const int n, lbdim, rbdim, bdim, gdim, cdim;
        const int nblock, ngrid, bblock, bgrid, gblock, ggrid, cblock, cgrid;
        const cumatrix *P, *q, *lb, *rb, *G, *h;
        cumatrix *L, *bd, *d, *dsq, *Gd;
        int *d_info;
        int d_worklen;
        double *d_work;
        void fG(char trans, const cumatrix* x, cumatrix* y) const;
        void kktfactor();
        void kktsolver(cumatrix* x, cumatrix* y, cumatrix* u) const;
};