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

__global__ void update_scaling(int len, double* d, double* t7u, const double* s, const double* z){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len){
        d[i] = std::sqrt(s[i] / z[i]);
        t7u[i] = std::sqrt(s[i] * z[i]);
    }
}

__global__ void update_sz1(int len, double* ds, double* var3, const double* d, const double*t7u, const double*v2a){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len){
        ds[i] = -t7u[i];
        var3[i] = -v2a[i] - d[i] * ds[i];
    }
}

__global__ void update_sz2(int len, double* ds, double* var3, const double* d, const double* t7u, const double* v2a, const double* s2, double vars1){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len){
        ds[i] = -t7u[i] + (vars1 - s2[i]) / t7u[i];
        var3[i] = -v2a[i] - d[i] * ds[i];
    }
}
__global__ void update_s2(int len, double* s2, const double* ds, const double* var3){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) s2[i] = ds[i] * var3[i];
}

__global__ void update_sz3(int len, double* s, double* z, double* ds, double* var3, double* d, double* t7u, double step){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len){
        ds[i] = t7u[i] * (step * ds[i] + 1.0);
        var3[i] = t7u[i] * (step * var3[i] + 1.0);
        s[i] = ds[i] * d[i];
        z[i] = var3[i] / d[i];
        ds[i] = std::sqrt(ds[i]);
        var3[i] = std::sqrt(var3[i]);
        d[i] *= ds[i] / var3[i];
        t7u[i] = ds[i] * var3[i];
    }
}


qp1::qp1(const CublasHandle& _cublas_handle, const CusolverHandle& _cusolver_handle, const matrix* P, const matrix* q, const matrix* lb, const matrix* rb, const matrix* G, const matrix* h):
    cublas_handle(_cublas_handle.handle),
    cusolver_handle(_cusolver_handle.handle),
    n(P->nrows), lbdim(lb?n:0), rbdim(rb?n:0),
    bdim(lbdim+rbdim), gdim(G?G->nrows:0),
    cdim(bdim+gdim),
    nblock(MIN(n, 1024)), ngrid((n<=1024)?1:((n+1023)/1024)),
    gblock(MIN(gdim, 1024)), ggrid((gdim<=1024)?1:((gdim+1023)/1024)),
    cblock(MIN(cdim, 1024)), cgrid((cdim<=1024)?1:((cdim+1023)/1024)), 
    P(P), q(q), lb(lb), rb(rb), G(G), h(h),
    L(new matrix(n, n)),
    d(new matrix(cdim, 1)),
    dsq(new matrix(cdim, 1)),
    Gd(new matrix(gdim, n)){
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, L->begin, n, &d_worklen));
        cudaMalloc(reinterpret_cast<void **>(&d_work), d_worklen);
    }


qp1::~qp1(){
    cudaFree(d_info);
    cudaFree(d_work);
}

void qp1::fG(char trans, const matrix* x, matrix* y) const{
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


void qp1::kktfactor(){
    const double dbl1=1.0;
    
    scal_d<<<cblock, cgrid>>>(cdim, dsq->begin, d->begin);
    if(gdim) scal_G<<<dim3(gblock, nblock, 1), dim3(ggrid, ngrid, 1)>>>(gdim, n, Gd->begin, G->begin, d->begin+bdim);
    cudaDeviceSynchronize();
    int mode = (lbdim?1:0) + (rbdim?1:0);
    if(mode == 2) cuda_add<<<nblock, ngrid>>>(n, dsq->begin, dsq->begin+n, 1.0);
    cudaDeviceSynchronize();
    
    if(mode==0) update_L0<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin);
    else update_L1<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin, dsq->begin);
    cudaDeviceSynchronize();
    cublasDsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, gdim, &dbl1, Gd->begin, gdim, &dbl1, L->begin, n);
    cudaDeviceSynchronize();
    
    int info;
    CUSOLVER_CHECK(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, L->begin, n, d_work, d_worklen, d_info));
    cudaDeviceSynchronize();
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(info < 0){
        std::cout << "Cholesky decomposition failed, the status code is" << info << std::endl;
        exit(1);
    }
}

void qp1::kktsolver(matrix* x, matrix* z, matrix* u) const{
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


matrix* qp1::solve(){
    const int MAXITERS=100;
    const double dbl1=1.0;
    const double EXPON=3.0, STEP=0.99, ABSTOL=1e-7, RELTOL=1e-6, FEASTOL=1e-7;
    int iters;
    double resx, resz, resx0, resz0, nrms;
    double f0, ts, tz, tt, temp, step;
    double pcost, dcost, pres, dres, gap, relgap;
    double sigma, mu, vars1, dsvar3;
    
    matrix *bh = new matrix(cdim, 1);
    if(lbdim) cuda_copy<<<nblock, ngrid>>>(lbdim, bh->begin, lb->begin, -1.0);
    if(rbdim) cuda_copy<<<nblock, ngrid>>>(rbdim, bh->begin+lbdim, rb->begin);
    if(gdim) cuda_copy<<<gblock, ggrid>>>(gdim, bh->begin+bdim, h->begin);
    cudaDeviceSynchronize();
    
    resx0 = q ? MAX(1.0, q->nrm2()) : 1.0;
    resz0 = cdim ? MAX(1.0, bh->nrm2()) : 1.0;
    
    d->fill(1.0);
    kktfactor();
    matrix *x = new matrix(n, 1);
    if(q) x->copy(q->begin, -1.0);
    else x->fill(0.0);
    matrix *z = new matrix(cdim, 1);
    z->copy(bh->begin);
    matrix *u = new matrix(cdim, 1);
    cudaDeviceSynchronize();
    kktsolver(x, z, u);
    matrix *s = new matrix(cdim, 1);
    s->copy(z->begin, -1.0);
    
    nrms = z->nrm2();
    tz = -z->min();
    cudaDeviceSynchronize();
    ts = -s->min();
    if(ts >= -1e-8 * MAX(nrms, 1.0)){temp = ts + 1.0; s->add(temp);}
    if(tz >= -1e-8 * MAX(nrms, 1.0)){temp = tz + 1.0; z->add(temp);}
    cudaDeviceSynchronize();
    gap = z->dot(s->begin);
    
    matrix *rx = new matrix(n, 1);
    matrix *v2a = new matrix(cdim, 1);
    matrix *dx = new matrix(n, 1);
    dx->copy(x->begin);
    matrix *var3 = new matrix(cdim, 1);
    matrix *ds = new matrix(cdim, 1);
    matrix *s2 = new matrix(cdim, 1);
    matrix *t7u = new matrix(cdim, 1);
    update_scaling<<<cblock, cgrid>>>(cdim, d->begin, t7u->begin, s->begin, z->begin);
    for(iters=0; iters<MAXITERS; iters++){
        
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
        
        v2a->copy(s->begin);
        cudaDeviceSynchronize();
        v2a->add(bh->begin, -1.0);
        cudaDeviceSynchronize();
        fG('N', x, v2a);
        resz = v2a->nrm2();
        
        pcost = f0;
        dcost = f0 + z->dot(v2a->begin) - gap;
        if(pcost<0.0) relgap = - gap / pcost;
        else if(dcost>0.0) relgap = gap / dcost;
        else relgap = 1.0;
        pres = resz / resz0;
        dres = resx / resx0;
        mu = gap / cdim;
        if(pres<=FEASTOL && dres<= FEASTOL && (gap<=ABSTOL || relgap<=RELTOL)) return x;
        kktfactor();
        
        update_sz1<<<cblock, cgrid>>>(cdim, ds->begin, var3->begin, d->begin, t7u->begin, v2a->begin);
        dx->copy(rx->begin, -1.0);
        cudaDeviceSynchronize();
        kktsolver(dx, var3, u);
        ds->add(var3->begin, -1.0);
        cudaDeviceSynchronize();
        update_s2<<<cblock, cgrid>>>(cdim, s2->begin, ds->begin, var3->begin);
        cudaDeviceSynchronize();
        dsvar3 = s2->sum();
        ds->divide(t7u->begin);
        var3->divide(t7u->begin);
        cudaDeviceSynchronize();
        ts = -ds->min();
        tz = -var3->min();
        tt = MAX(0.0, MAX(ts, tz));
        step = (tt==0.0)?1.0:MIN(1.0, 1.0/tt);
        sigma = pow(MIN(1.0, MAX(0.0, 1.0-step+dsvar3/gap*step*step)), EXPON);
        vars1 = sigma * mu;
        
        update_sz2<<<cblock, cgrid>>>(cdim, ds->begin, var3->begin, d->begin, t7u->begin, v2a->begin, s2->begin, vars1);
        dx->copy(rx->begin, -1.0);
        cudaDeviceSynchronize();
        kktsolver(dx, var3, u);
        ds->add(var3->begin, -1.0);
        cudaDeviceSynchronize();
        dsvar3 = ds->dot(var3->begin);
        ds->divide(t7u->begin);
        var3->divide(t7u->begin);
        cudaDeviceSynchronize();
        ts = -ds->min();
        tz = -var3->min();
        tt = MAX(0.0, MAX(ts, tz));
        step = (tt==0.0)?1.0:MIN(1.0, STEP/tt);
        x->add(dx->begin, step);
        update_sz3<<<cblock, cgrid>>>(cdim, s->begin, z->begin, ds->begin, var3->begin, d->begin, t7u->begin, step);
        cudaDeviceSynchronize();
        gap = s->dot(z->begin);
    }
    std::cerr << "Warning: Max number of iterations reached" << std::endl;
    return x;
}




qp2::qp2(const CublasHandle& _cublas_handle, const CusolverHandle& _cusolver_handle, const matrix* P, const matrix* q, const matrix* lb, const matrix* rb, const matrix* G, const matrix* h):
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
    L(new matrix(n, n)),
    bd(new matrix(bdim, 1)),
    d(new matrix(cdim, 1)),
    dsq(new matrix(cdim, 1)),
    Gd(new matrix(gdim, n)){
        cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, L->begin, n, &d_worklen));
        cudaMalloc(reinterpret_cast<void **>(&d_work), d_worklen);
    }

qp2::~qp2(){
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

void qp2::fG(char trans, const matrix* x, matrix* y) const{
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


void qp2::kktfactor(){
    double bdb=0.0;
    const double dbl1=1.0;
    
    scal_d<<<cblock, cgrid>>>(cdim, dsq->begin, d->begin);
    if(gdim) scal_G<<<dim3(gblock, nblock, 1), dim3(ggrid, ngrid, 1)>>>(gdim, n, Gd->begin, G->begin, d->begin+bdim);
    cudaDeviceSynchronize();
    
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
    
    if(mode==0) update_L0<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin);
    else update_L2<<<dim3(nblock, nblock, 1), dim3(ngrid, ngrid, 1)>>>(n, L->begin, P->begin, bd->begin, dsq->begin, bdb);
    cudaDeviceSynchronize();
    cublasDsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, gdim, &dbl1, Gd->begin, gdim, &dbl1, L->begin, n);
    cudaDeviceSynchronize();
    
    int info;
    CUSOLVER_CHECK(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, L->begin, n, d_work, d_worklen, d_info));
    cudaDeviceSynchronize();
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(info < 0){
        std::cout << "Cholesky decomposition failed, the status code is" << info << std::endl;
        exit(1);
    }
}

void qp2::kktsolver(matrix* x, matrix* z, matrix* u) const{
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


matrix* qp2::solve(){
    const int MAXITERS=100;
    const double dbl1=1.0;
    const double EXPON=3.0, STEP=0.99, ABSTOL=1e-7, RELTOL=1e-6, FEASTOL=1e-7;
    int iters;
    double resx, resz, resx0, resz0, nrms;
    double f0, ts, tz, tt, temp, step;
    double pcost, dcost, pres, dres, gap, relgap;
    double sigma, mu, vars1, dsvar3;
    
    matrix *bh = new matrix(cdim, 1);
    if(bdim) cuda_fill<<<bblock, bgrid>>>(bdim, bh->begin, 0.0);
    if(gdim) cuda_copy<<<gblock, ggrid>>>(gdim, bh->begin+bdim, h->begin);
    
    resx0 = q ? MAX(1.0, q->nrm2()) : 1.0;
    resz0 = gdim ? MAX(1.0, h->nrm2()) : 1.0;
    cudaDeviceSynchronize();
    
    d->fill(1.0);
    kktfactor();
    matrix *x = new matrix(n, 1);
    if(q) x->copy(q->begin, -1.0);
    else x->fill(0.0);
    matrix *z = new matrix(cdim, 1);
    z->copy(bh->begin);
    matrix *u = new matrix(cdim, 1);
    cudaDeviceSynchronize();
    kktsolver(x, z, u);
    matrix *s = new matrix(cdim, 1);
    s->copy(z->begin, -1.0);
    
    nrms = z->nrm2();
    tz = -z->min();
    cudaDeviceSynchronize();
    ts = -s->min();
    if(ts >= -1e-8 * MAX(nrms, 1.0)){temp = ts + 1.0; s->add(temp);}
    if(tz >= -1e-8 * MAX(nrms, 1.0)){temp = tz + 1.0; z->add(temp);}
    cudaDeviceSynchronize();
    gap = z->dot(s->begin);
    
    matrix *rx = new matrix(n, 1);
    matrix *v2a = new matrix(cdim, 1);
    matrix *dx = new matrix(n, 1);
    dx->copy(x->begin);
    matrix *var3 = new matrix(cdim, 1);
    matrix *ds = new matrix(cdim, 1);
    matrix *s2 = new matrix(cdim, 1);
    matrix *t7u = new matrix(cdim, 1);
    update_scaling<<<cblock, cgrid>>>(cdim, d->begin, t7u->begin, s->begin, z->begin);
    for(iters=0; iters<MAXITERS; iters++){
        
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
        
        v2a->copy(s->begin);
        cudaDeviceSynchronize();
        if(gdim) cuda_add<<<gblock, ggrid>>>(gdim, v2a->begin+bdim, h->begin, -1.0);
        cudaDeviceSynchronize();
        fG('N', x, v2a);
        resz = v2a->nrm2();
        
        pcost = f0;
        dcost = f0 + z->dot(v2a->begin) - gap;
        if(pcost<0.0) relgap = - gap / pcost;
        else if(dcost>0.0) relgap = gap / dcost;
        else relgap = 1.0;
        pres = resz / resz0;
        dres = resx / resx0;
        mu = gap / cdim;
        if(pres<=FEASTOL && dres<= FEASTOL && (gap<=ABSTOL || relgap<=RELTOL)) return x;
        kktfactor();
        
        update_sz1<<<cblock, cgrid>>>(cdim, ds->begin, var3->begin, d->begin, t7u->begin, v2a->begin);
        dx->copy(rx->begin, -1.0);
        cudaDeviceSynchronize();
        kktsolver(dx, var3, u);
        ds->add(var3->begin, -1.0);
        cudaDeviceSynchronize();
        update_s2<<<cblock, cgrid>>>(cdim, s2->begin, ds->begin, var3->begin);
        cudaDeviceSynchronize();
        dsvar3 = s2->sum();
        ds->divide(t7u->begin);
        var3->divide(t7u->begin);
        cudaDeviceSynchronize();
        ts = -ds->min();
        tz = -var3->min();
        tt = MAX(0.0, MAX(ts, tz));
        step = (tt==0.0)?1.0:MIN(1.0, 1.0/tt);
        sigma = pow(MIN(1.0, MAX(0.0, 1.0-step+dsvar3/gap*step*step)), EXPON);
        vars1 = sigma * mu;
        
        update_sz2<<<cblock, cgrid>>>(cdim, ds->begin, var3->begin, d->begin, t7u->begin, v2a->begin, s2->begin, vars1);
        dx->copy(rx->begin, -1.0);
        cudaDeviceSynchronize();
        kktsolver(dx, var3, u);
        ds->add(var3->begin, -1.0);
        cudaDeviceSynchronize();
        dsvar3 = ds->dot(var3->begin);
        ds->divide(t7u->begin);
        var3->divide(t7u->begin);
        cudaDeviceSynchronize();
        ts = -ds->min();
        tz = -var3->min();
        tt = MAX(0.0, MAX(ts, tz));
        step = (tt==0.0)?1.0:MIN(1.0, STEP/tt);
        x->add(dx->begin, step);
        update_sz3<<<cblock, cgrid>>>(cdim, s->begin, z->begin, ds->begin, var3->begin, d->begin, t7u->begin, step);
        cudaDeviceSynchronize();
        gap = s->dot(z->begin);
    }
    std::cerr << "Warning: Max number of iterations reached" << std::endl;
    return x;
}