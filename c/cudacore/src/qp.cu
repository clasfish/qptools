#include <iostream>
#include "matrix_base.cuh"
#include "matrix_util.cuh"
#include "qp.cuh"


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

__global__ void update_L2(int n, double* L, const double* P, const double* bd, const double* dsq, double bdb){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<n){
        if(j<i) L[i+j*n] = P[i+j*n] + bdb - bd[i] - bd[j];
        else if(j==i) L[i+j*n] = P[i+j*n] + dsq[i] + bdb - 2 * bd[i];
    }
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

// ------------------------cuqp1
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
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, L->begin, n, &d_worklen));
        cudaMalloc(reinterpret_cast<void **>(&d_work), d_worklen);
    }


cuqp1::~cuqp1(){
    cudaFree(d_info);
    cudaFree(d_work);
}

void cuqp1::fG(char trans, const cumatrix* x, cumatrix* y) const{
    const double dbl1=1.0;
    if(trans == 'N'){
        if(lbdim) cuda_add<<<nblock, ngrid>>>(n, y->begin, x->begin, -1.0);
        if(rbdim) cuda_add<<<nblock, ngrid>>>(n, y->begin+lbdim, x->begin, 1.0);
        if(gdim) cublasDgemv(cublas_handle, CUBLAS_OP_N, gdim, n, &dbl1, G->begin, gdim, x->begin, 1, &dbl1, y->begin+bdim, 1);
        cudaDeviceSynchronize();
    }else{
        if(lbdim){cuda_add<<<nblock, ngrid>>>(n, y->begin, x->begin, -1.0); cudaDeviceSynchronize();}
        if(lbdim){cuda_add<<<nblock, ngrid>>>(n, y->begin, x->begin+lbdim, 1.0); cudaDeviceSynchronize();}
        if(gdim){cublasDgemv(cublas_handle, CUBLAS_OP_T, gdim, n, &dbl1, G->begin, gdim, x->begin+bdim, 1, &dbl1, y->begin, 1);cudaDeviceSynchronize();}
    }
}


void cuqp1::kktfactor(){
    const double dbl1=1.0;
    //dsq = d{-2}, Gd = d{-1}G
    scal_d<<<cblock, cgrid>>>(cdim, dsq->begin, d->begin);
    if(gdim) scal_G<<<dim3(gblock, nblock, 1), dim3(ggrid, ngrid, 1)>>>(gdim, n, Gd->begin, G->begin, d->begin+bdim);
    cudaDeviceSynchronize();
    int mode = (lbdim?1:0) + (rbdim?1:0);
    if(mode == 2) cuda_add<<<nblock, ngrid>>>(n, dsq->begin, dsq->begin+n, 1.0);
    cudaDeviceSynchronize();
    // L = Gd{T}Gd + P  
    if(mode==0) update_L0<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin);
    else update_L1<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin, dsq->begin);
    cudaDeviceSynchronize();
    cublasDsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, gdim, &dbl1, Gd->begin, gdim, &dbl1, L->begin, n);
    cudaDeviceSynchronize();
    // potrf
    int info;
    CUSOLVER_CHECK(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, L->begin, n, d_work, d_worklen, d_info));
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
    gap = z->dot(s->begin);
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
    std::cerr << "Warning: Max number of iterations reached" << std::endl;
    return x;
}


// ------------------------cuqp2

cuqp2::cuqp2(const CublasHandle& _cublas_handle, const CusolverHandle& _cusolver_handle, const cumatrix* P, const cumatrix* q, const cumatrix* lb, const cumatrix* rb, const cumatrix* G, const cumatrix* h):
    cublas_handle(_cublas_handle.handle),
    cusolver_handle(_cusolver_handle.handle),
    n(P->nrows), lbdim(lb?n:0), rbdim(rb?n:0),
    bdim(lbdim+rbdim), gdim(G?G->nrows:0),
    cdim(bdim+gdim),
    nblock(MIN(n, 1024)), ngrid((n<=1024)?1:((n+1023)/1024)),
    bblock(MIN(bdim, 1024)), bgrid((bdim<=1024)?1:((bdim+1023)/1024)),
    gblock(MIN(gdim, 1024)), ggrid((gdim<=1024)?1:((gdim+1023)/1024)),
    cblock(MIN(cdim, 1024)), cgrid((cdim<=1024)?1:((cdim+1023)/1024)), 
    P(P), q(q), lb(lb), rb(rb), G(G), h(h),
    L(new cumatrix(n, n)),
    bd(new cumatrix(bdim, 1)),
    d(new cumatrix(cdim, 1)),
    dsq(new cumatrix(cdim, 1)),
    Gd(new cumatrix(gdim, n)){
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, L->begin, n, &d_worklen));
        cudaMalloc(reinterpret_cast<void **>(&d_work), d_worklen);
    }

cuqp2::~cuqp2(){
    cudaFree(d_info);
    cudaFree(d_work);
}

cusolverStatus_t
cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle,
                 cublasFillMode_t uplo,
                 int n,
                 double *A,
                 int lda,
                 int *Lwork );

void cuqp2::fG(char trans, const cumatrix* x, cumatrix* y) const{
    double s;
    const double dbl1 = 1.0;
    if(trans == 'N'){
        s = x->sum();
        if(lbdim){
            cuda_add<<<nblock, ngrid>>>(n, y->begin, x->begin, -1.0);
            cudaDeviceSynchronize();
            cuda_add<<<nblock, ngrid>>>(n, y->begin, lb->begin, s);
        }
        if(rbdim){
            cuda_add<<<nblock, ngrid>>>(n, y->begin+lbdim, x->begin, 1.0);
            cudaDeviceSynchronize();
            cuda_add<<<nblock, ngrid>>>(n, y->begin+lbdim, rb->begin, -s);
        }
        if(gdim) cublasDgemv(cublas_handle, CUBLAS_OP_N, gdim, n, &dbl1, G->begin, gdim, x->begin, 1, &dbl1, y->begin+bdim, 1);
        cudaDeviceSynchronize();
    }else{
        s = 0.0;
        if(lbdim){
            cuda_add<<<nblock, ngrid>>>(n, y->begin, x->begin, -1.0);
            s += lb->dot(x->begin);
            cudaDeviceSynchronize();
        }
        if(rbdim){
            cuda_add<<<nblock, ngrid>>>(n, y->begin, x->begin+lbdim, 1.0);
            s -= rb->dot(x->begin+lbdim);
            cudaDeviceSynchronize();
        }
        if(bdim){y->add(s); cudaDeviceSynchronize();}
        if(gdim){cublasDgemv(cublas_handle, CUBLAS_OP_T, gdim, n, &dbl1, G->begin, gdim, x->begin+bdim, 1, &dbl1, y->begin, 1); cudaDeviceSynchronize();}
    }
}


void cuqp2::kktfactor(){
    double bdb=0.0;
    const double dbl1=1.0;
    //dsq = d{-2}, Gd = d{-1}G
    scal_d<<<cblock, cgrid>>>(cdim, dsq->begin, d->begin);
    if(gdim) scal_G<<<dim3(gblock, nblock, 1), dim3(ggrid, ngrid, 1)>>>(gdim, n, Gd->begin, G->begin, d->begin+bdim);
    cudaDeviceSynchronize();
    //bd
    if(lbdim){
        cuda_fmcopy<<<nblock, ngrid>>>(n, bd->begin, lb->begin, dsq->begin);
        cudaDeviceSynchronize();
        bdb += lb->dot(bd->begin);
    }
    if(rbdim){
        cuda_fmcopy<<<nblock, ngrid>>>(n, bd->begin+lbdim, rb->begin, dsq->begin+lbdim);
        cudaDeviceSynchronize();
        bdb += rb->dot(bd->begin+lbdim);
    }
    int mode = (lbdim?1:0) + (rbdim?1:0);
    if(mode == 2){
        cuda_add<<<nblock, ngrid>>>(n, bd->begin, bd->begin+n, 1.0);
        cuda_add<<<nblock, ngrid>>>(n, dsq->begin, dsq->begin+n, 1.0);
        cudaDeviceSynchronize();
    }
    // L = Gd{T}Gd + P
    if(mode==0) update_L0<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin);
    else update_L2<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin, bd->begin, dsq->begin, bdb);
    cudaDeviceSynchronize();
    cublasDsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, gdim, &dbl1, Gd->begin, gdim, &dbl1, L->begin, n);
    cudaDeviceSynchronize();
    // potrf
    int info;
    CUSOLVER_CHECK(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, L->begin, n, d_work, d_worklen, d_info));
    cudaDeviceSynchronize();
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(info < 0){
        std::cout << "Cholesky decomposition failed, the status code is" << info << std::endl;
        exit(1);
    }
}

void cuqp2::kktsolver(cumatrix* x, cumatrix* z, cumatrix* u) const{
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


cumatrix* cuqp2::solve(){
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
    if(bdim) cuda_fill<<<bblock, bgrid>>>(bdim, bh->begin, 0.0);
    if(gdim) cuda_copy<<<gblock, ggrid>>>(gdim, bh->begin+bdim, h->begin);
    // res0
    resx0 = q ? MAX(1.0, q->nrm2()) : 1.0;
    resz0 = gdim ? MAX(1.0, h->nrm2()) : 1.0;
    cudaDeviceSynchronize();
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
    cudaDeviceSynchronize();
    gap = z->dot(s->begin);
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
        // rx = Px + q +G'z
        rx->fill(0.0);
        cudaDeviceSynchronize();
        cublasDsymv(cublas_handle, CUBLAS_FILL_MODE_LOWER, n, &dbl1, P->begin, n, x->begin, 1, &dbl1, rx->begin, 1);
        cudaDeviceSynchronize();
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
        if(gdim) cuda_add<<<gblock, ggrid>>>(gdim, rz->begin+bdim, h->begin, -1.0);
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
    std::cerr << "Warning: Max number of iterations reached" << std::endl;
    return x;
}