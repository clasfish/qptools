#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

class qp1{
    public:
        qp1(const CublasHandle& cublas_handle, const CusolverHandle& cusolver_handle, const cumatrix* P, const cumatrix* q, const cumatrix* lb, const cumatrix* rb, const cumatrix* G, const cumatrix* h);
        cumatrix* solve();
    private:
        cublasHandle_t cublas_handle;
        cusolverDnHandle_t cusolver_handle;
        cusolverDnParams_t cusolver_params;
        const int n, lbdim, rbdim, bdim, gdim, cdim;
        const int nblock, ngrid, gblock, ggrid, cblock, cgrid;
        const cumatrix *P, *q, *lb, *rb, *G, *h;
        cumatrix *L, *d, *dsq, *Gd;
        int *d_info;
        size_t d_worklen, h_worklen;
        void *d_work, *h_work;
        void fG(char trans, const cumatrix* x, cumatrix* y) const;
        void kktfactor();
        void kktsolver(cumatrix* x, cumatrix* y, cumatrix* u) const;
};