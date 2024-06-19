#include "cumatrix_base.h"
#include "cumatrix_util.h"
#include "cuqp.h"
#include "util.h"
#include <thrust/reduce.h>
#define MIN(i, j) (((i) < (j)) ? (i) : (j))
#define MAX(i, j) (((i) > (j)) ? (i) : (j))

cuqp1::cuqp1(const CublasHandle& _cublas_handle, const CusolverHandle& _cusolver_handle, const cumatrix* P, const cumatrix* q, const cumatrix* lb, const cumatrix* rb, const cumatrix* G, const cumatrix* h):
    cublas_handle(_cublas_handle.handle),
    cusolver_handle(_cusolver_handle.handle),
    n(P->nrows), lbdim(lb?n:0), rbdim(rb?n:0),
    bdim(lbdim+rbdim), gdim(G?G->nrows:0),
    cdim(bdim+gdim),
    nblock(MIN(n, 1024)), ngrid((n<=1024)?1:((n+1023)/1024)),
    gblock(MIN(gdim, 1024)), ggrid((gdim<=1024)?1:((gdim+1023)/1024)),
    cblock(MIN(cdim, 1024)), cgrid((cdim<=1024)?1:((cdim+1023)/1024)), 
    P(P), q(q), lb(lb), rb(rb), G(G), h(h),
    L(new cumatrix(n, n)),
    d(new cumatrix(cdim, 1)),
    dsq(new cumatrix(cdim, 1)),
    Gd(new cumatrix(gdim, n)){
        cusolverDnCreateParams(&cusolver_params);
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
        cusolverDnXpotrf_bufferSize(cusolver_handle, cusolver_params, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L->begin, n, CUDA_R_64F, &d_worklen, &h_worklen);
        cudaMalloc(reinterpret_cast<void **>(&d_work), d_worklen);
        if(h_worklen > 0) h_work = reinterpret_cast<void *>(malloc(h_worklen));
        else h_work = nullptr;
    }


void cuqp1::fG(char trans, const cumatrix* x, cumatrix* y) const{
    const double dbl1=1.0;
    if(trans == 'N'){
        if(lbdim) cuda_add<<<nblock, ngrid>>>(n, y->begin, x->begin, -1.0);
        if(rbdim) cuda_add<<<nblock, ngrid>>>(n, y->begin+lbdim, x->begin, 1.0);
        if(gdim) cublasDgemv(cublas_handle, CUBLAS_OP_N, gdim, n, &dbl1, G->begin, gdim, x->begin, 1, &dbl1, y->begin+bdim, 1);
    }else{
        if(lbdim) cuda_add<<<nblock, ngrid>>>(n, y->begin, x->begin, -1.0);
        cudaDeviceSynchronize();
        if(lbdim) cuda_add<<<nblock, ngrid>>>(n, y->begin, x->begin+lbdim, 1.0);
        cudaDeviceSynchronize();
        if(gdim) cublasDgemv(cublas_handle, CUBLAS_OP_T, gdim, n, &dbl1, G->begin, gdim, x->begin+bdim, 1, &dbl1, y->begin, 1);
    }
}

__global__ void scal_d(int len, double* dsq, const double* d){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) dsq[i] = 1 / (d[i] * d[i]);
}

__global__ void scal_G(int gdim, int n, double* Gd, const double* G, const double* d){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if((i<gdim) && (j<n)) Gd[i+j*gdim] = G[i+j*gdim] / d[i];
}

__global__ void update_L0(int n, double* L, const double* P){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if((i<n) && (j<=i)) L[i+j*n] = P[i+j*n];
}

__global__ void update_L1(int n, double* L, const double* P, const double* dsq){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<n){
        if(j<i) L[i+j*n] = P[i+j*n];
        else if(j==i) L[i+j*n] = P[i+j*n] + dsq[i];
    }
}

__global__ void update_L2(int n, double* L, const double* P, const double* dsq){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<n){
        if(j<i) L[i+j*n] = P[i+j*n];
        else if(j==i) L[i+j*n] = P[i+j*n] + dsq[i] + dsq[n+i];
    }
}




void cuqp1::kktfactor(){
    const double dbl1=1.0;
    //dsq = d{-2}, Gd = d{-1}G
    scal_d<<<cblock, cgrid>>>(cdim, dsq->begin, d->begin);
    if(gdim) scal_G<<<dim3(gblock, nblock, 1), dim3(ggrid, ngrid, 1)>>>(gdim, n, Gd->begin, G->begin, d->begin+bdim);
    cudaDeviceSynchronize();
    // L = Gd{T}Gd + P
    int mode = (lbdim?1:0) + (rbdim?1:0);
    if(mode==0) update_L0<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin);
    else if(mode==1) update_L1<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin, dsq->begin);
    else update_L2<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin, dsq->begin);
    cudaDeviceSynchronize();
    cublasDsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, gdim, &dbl1, Gd->begin, gdim, &dbl1, L->begin, n);
    // potrf
    int info;;
    CUSOLVER_CHECK(cusolverDnXpotrf(cusolver_handle, cusolver_params, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L->begin, n, CUDA_R_64F, d_work, d_worklen, h_work, h_worklen, d_info));
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(info < 0){
        std::cout << "Cholesky decomposition failed, the status code is" << info << std::endl;
        exit(1);
    }
}

void cuqp1::kktsolver(cumatrix* x, cumatrix* z, cumatrix* u) const{
    z->divide(d->begin);
    cudaDeviceSynchronize();
    u->copy(z->begin);
    cudaDeviceSynchronize();
    z->divide(d->begin);
    cudaDeviceSynchronize();
    fG('T', z, x);
    cublasDtrsv(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, L->begin, n, x->begin, 1);
    cublasDtrsv(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, L->begin, n, x->begin, 1);
    z->fill(0.0);
    cudaDeviceSynchronize();
    fG('N', x, z);
    z->divide(d->begin);
    cudaDeviceSynchronize();
    z->add(u->begin, -1.0);
    cudaDeviceSynchronize();
}

__global__ void update_scaling(int len, double* d, double* lambda, const double* s, const double* z){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len){
        d[i] = std::sqrt(s[i] / z[i]);
        lambda[i] = std::sqrt(s[i] * z[i]);
    }
}

__global__ void update_sz1(int len, double* ds, double* dz, const double* d, const double*lambda, const double*rz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len){
        ds[i] = -lambda[i];
        dz[i] = -rz[i] - d[i] * ds[i];
    }
}

__global__ void update_sz2(int len, double* ds, double* dz, const double* d, const double* lambda, const double* rz, const double* s2, double sigmamu){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len){
        ds[i] = -lambda[i] + (sigmamu - s2[i]) / lambda[i];
        dz[i] = -rz[i] - d[i] * ds[i];
    }
}
__global__ void update_s2(int len, double* s2, const double* ds, const double* dz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) s2[i] = ds[i] * dz[i];
}

__global__ void update_sz3(int len, double* s, double* z, double* ds, double* dz, double* d, double* lambda, double step){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len){
        ds[i] = lambda[i] * (step * ds[i] + 1.0);
        dz[i] = lambda[i] * (step * dz[i] + 1.0);
        s[i] = ds[i] * d[i];
        z[i] = dz[i] / d[i];
        ds[i] = std::sqrt(ds[i]);
        dz[i] = std::sqrt(dz[i]);
        d[i] *= ds[i] / dz[i];
        lambda[i] = ds[i] * dz[i];
    }
}
cumatrix* cuqp1::solve(){
    const int MAXITERS=100;
    const double dbl1=1.0;
    const double EXPON=3.0, STEP=0.99, ABSTOL=1e-7, RELTOL=1e-6, FEASTOL=1e-7;
    int iters;
    double resx, resz, resx0, resz0, nrms;
    double f0, ts, tz, tt, temp, step;
    double pcost, dcost, pres, dres, gap, relgap;
    double sigma, mu, sigmamu, dsdz;
    // bh
    cumatrix *bh = new cumatrix(cdim, 1);
    if(lbdim) cuda_copy<<<nblock, ngrid>>>(lbdim, bh->begin, lb->begin, -1.0);
    if(rbdim) cuda_copy<<<nblock, ngrid>>>(rbdim, bh->begin+lbdim, rb->begin);
    if(gdim) cuda_copy<<<gblock, ggrid>>>(gdim, bh->begin+bdim, h->begin);
    cudaDeviceSynchronize();
    // res0
    resx0 = q ? MAX(1.0, q->nrm2()) : 1.0;
    resz0 = cdim ? MAX(1.0, bh->nrm2()) : 1.0;
    // initialize
    d->fill(1.0);
    kktfactor();
    cumatrix *x = new cumatrix(n, 1);
    if(q) x->copy(q->begin, -1.0);
    else x->fill(0.0);
    cumatrix *z = new cumatrix(cdim, 1);
    z->copy(bh->begin);
    cumatrix *u = new cumatrix(cdim, 1);
    cudaDeviceSynchronize();
    kktsolver(x, z, u);
    cumatrix *s = new cumatrix(cdim, 1);
    s->copy(z->begin, -1.0);
    // ts & tz
    nrms = z->nrm2();
    tz = -z->min();
    cudaDeviceSynchronize();
    ts = -s->min();
    if(ts >= -1e-8 * MAX(nrms, 1.0)){temp = ts + 1.0; s->add(temp);}
    if(tz >= -1e-8 * MAX(nrms, 1.0)){temp = tz + 1.0; z->add(temp);}
    cublasDdot(cublas_handle, cdim, s->begin, 1, z->begin, 1, &gap);
    // steps
    cumatrix *rx = new cumatrix(n, 1);
    cumatrix *rz = new cumatrix(cdim, 1);
    cumatrix *dx = new cumatrix(n, 1);
    dx->copy(x->begin);
    cumatrix *dz = new cumatrix(cdim, 1);
    cumatrix *ds = new cumatrix(cdim, 1);
    cumatrix *s2 = new cumatrix(cdim, 1);
    cumatrix *lambda = new cumatrix(cdim, 1);
    update_scaling<<<cblock, cgrid>>>(cdim, d->begin, lambda->begin, s->begin, z->begin);
    for(iters=0; iters<MAXITERS; iters++){
        if(iters==20) return nullptr;
        // rx = Px + q +G'z
        rx->fill(0.0);
        cudaDeviceSynchronize();
        cublasDsymv(cublas_handle, CUBLAS_FILL_MODE_LOWER, n, &dbl1, P->begin, n, x->begin, 1, &dbl1, rx->begin, 1);
        if(q){
            f0 += 0.5 * x->dot(rx->begin);
            rx->add(q->begin, 1.0);
            cudaDeviceSynchronize();
        }
        fG('T', z, rx);
        resx = rx->nrm2();
        // rz = Gx + s - h
        rz->copy(s->begin);
        cudaDeviceSynchronize();
        rz->add(bh->begin, -1.0);
        cudaDeviceSynchronize();
        fG('N', x, rz);
        resz = rz->nrm2();
        // cost
        pcost = f0;
        dcost = f0 + z->dot(rz->begin) - gap;
        if(pcost<0.0) relgap = - gap / pcost;
        else if(dcost>0.0) relgap = gap / dcost;
        else relgap = 1.0;
        pres = resz / resz0;
        dres = resx / resx0;
        mu = gap / cdim;
        if(pres<=FEASTOL && dres<= FEASTOL && (gap<=ABSTOL || relgap<=RELTOL)) return x;
        kktfactor();
        // f4.0
        update_sz1<<<cblock, cgrid>>>(cdim, ds->begin, dz->begin, d->begin, lambda->begin, rz->begin);
        dx->copy(rx->begin, -1.0);
        cudaDeviceSynchronize();
        kktsolver(dx, dz, u);
        ds->add(dz->begin, -1.0);
        cudaDeviceSynchronize();
        update_s2<<<cblock, cgrid>>>(cdim, s2->begin, ds->begin, dz->begin);
        cudaDeviceSynchronize();
        dsdz = s2->sum();
        ds->divide(lambda->begin);
        dz->divide(lambda->begin);
        cudaDeviceSynchronize();
        ts = -ds->min();
        tz = -dz->min();
        tt = MAX(0.0, MAX(ts, tz));
        step = (tt==0.0)?1.0:MIN(1.0, 1.0/tt);
        sigma = pow(MIN(1.0, MAX(0.0, 1.0-step+dsdz/gap*step*step)), EXPON);
        sigmamu = sigma * mu;
        // f4.1
        update_sz2<<<cblock, cgrid>>>(cdim, ds->begin, dz->begin, d->begin, lambda->begin, rz->begin, s2->begin, sigmamu);
        dx->copy(rx->begin, -1.0);
        cudaDeviceSynchronize();
        kktsolver(dx, dz, u);
        ds->add(dz->begin, -1.0);
        cudaDeviceSynchronize();
        dsdz = ds->dot(dz->begin);
        ds->divide(lambda->begin);
        dz->divide(lambda->begin);
        cudaDeviceSynchronize();
        ts = -ds->min();
        tz = -dz->min();
        tt = MAX(0.0, MAX(ts, tz));
        step = (tt==0.0)?1.0:MIN(1.0, STEP/tt);
        x->add(dx->begin, step);
        update_sz3<<<cblock, cgrid>>>(cdim, s->begin, z->begin, ds->begin, dz->begin, d->begin, lambda->begin, step);
        cudaDeviceSynchronize();
        gap = s->dot(z->begin);
    }
}

/*
        // update
        x->add(dx->begin, step);
        for(i=0; i<cdim; i++){
            ds->begin[i] = lambda->begin[i] * (step * ds->begin[i] + 1.0);
            dz->begin[i] = lambda->begin[i] * (step * dz->begin[i] + 1.0);
            s->begin[i] = ds->begin[i] * d->begin[i];
            z->begin[i] = dz->begin[i] / d->begin[i];
            ds->begin[i] = std::sqrt(ds->begin[i]);
            dz->begin[i] = std::sqrt(dz->begin[i]);
            d->begin[i] *= ds->begin[i] / dz->begin[i];
            lambda->begin[i] = ds->begin[i] * dz->begin[i];
        }
        gap = ddot_(&cdim, s->begin, &int1, z->begin, &int1);
    }
    std::cout << "Max number of iterations reached" << std::endl;
    return x;
}
*/