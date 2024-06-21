#include "matrix_blas.h"
#include "qp.h"
#include <cmath>
#include <iostream>


qp1::qp1(const matrix* P, const matrix* q, const matrix* lb, const matrix* rb, const matrix* G, const matrix* h):
    n(P->nrows), lbdim(lb?n:0), rbdim(rb?n:0),
    bdim(lbdim+rbdim), gdim(G?G->nrows:0),
    cdim(bdim+gdim),
    P(P), q(q), lb(lb), rb(rb), G(G), h(h),
    L(new matrix(n, n)),
    d(new matrix(cdim, 1)),
    dsq(new matrix(cdim, 1)),
    Gd(new matrix(gdim, n)){}

void qp1::fG(char trans, const matrix* x, matrix* y) const{
    const int int1=1;
    const double dblm=-1.0, dbl1=1.0;
    if(trans == 'N'){
        if(lbdim) daxpy_(&lbdim, &dblm, x->begin, &int1, y->begin, &int1);
        if(rbdim) daxpy_(&rbdim, &dbl1, x->begin, &int1, y->begin+lbdim, &int1);
        if(gdim) dgemv_("N", &gdim, &n, &dbl1, G->begin, &gdim, x->begin, &int1, &dbl1, y->begin+bdim, &int1);
    }else{
        if(lbdim) daxpy_(&lbdim, &dblm, x->begin, &int1, y->begin, &int1);
        if(rbdim) daxpy_(&rbdim, &dbl1, x->begin+lbdim, &int1, y->begin, &int1);
        if(gdim) dgemv_("T", &gdim, &n, &dbl1, G->begin, &gdim, x->begin+bdim, &int1, &dbl1, y->begin, &int1);
    }
}

void qp1::kktfactor(){
    int i, j, info, n1=n+1;
    const double dbl1=1.0;
    double *iter1;
    const double *iter01, *iter02, *iter03;
    // dsq = d{-2}
    for(i=0; i<cdim; i++) dsq->begin[i] = 1 / (d->begin[i] * d->begin[i]);
    // Gd = d{-1}G
    if(gdim){
        iter01 = G->begin;
        iter1 = Gd->begin;
        iter03 = d->begin + bdim;
        for(j=0; j<n; j++){
            iter02 = iter03;
            for(i=0; i<gdim; i++) *(iter1++) = *(iter01++) / *(iter02++);
        }
    }
    // L = Gd{T}Gd + P
    for(j=0; j<n; j++){
        iter1 = L->begin+(j*n+j);
        iter01 = P->begin+(j*n+j);
        for(i=j; i<n; i++) *(iter1++) = *(iter01++);
    }
    if(lbdim){
        iter1 = L->begin;
        iter01 = dsq->begin;
        for(i=0; i<n; i++, iter1+=n1, iter01++) *iter1 += *iter01;
    }
    if(rbdim){
        iter1 = L->begin;
        iter01 = dsq->begin+lbdim;
        for(i=0; i<n; i++, iter1+=n1, iter01++) *iter1 += *iter01;
    }
    if(gdim) dsyrk_("L", "T", &n, &gdim, &dbl1, Gd->begin, &gdim, &dbl1, L->begin, &n);
    dpotrf_("L", &n, L->begin, &n, &info);
}

void qp1::kktsolver(matrix* x, matrix* z, matrix* u) const{
    const int int1=1;   
    z->divide(d->begin);
    u->copy(z->begin);
    z->divide(d->begin);
    fG('T', z, x);
    dtrsv_("L", "N", "N", &n, L->begin, &n, x->begin, &int1);
    dtrsv_("L", "T", "N", &n, L->begin, &n, x->begin, &int1);
    z->fill(0.0);
    fG('N', x, z);
    z->divide(d->begin);
    z->add(u->begin, -1.0);
}

matrix* qp1::solve(){
    const int int1=1;
    const int MAXITERS=100;
    const double dbl1=1.0;
    const double EXPON=3.0, STEP=0.99, ABSTOL=1e-7, RELTOL=1e-6, FEASTOL=1e-7;
    int iters, i;
    double resx, resz, resx0, resz0, nrms;
    double f0, ts, tz, tt, temp, step;
    double pcost, dcost, pres, dres, gap, relgap;
    double sigma, mu, sigmamu, dsdz;
    // bh
    matrix *bh = new matrix(cdim, 1);
    if(lbdim) for(i=0; i<lbdim; i++) bh->begin[i] = -lb->begin[i];
    if(rbdim) std::copy(rb->begin, rb->begin+rbdim, bh->begin+lbdim);
    if(gdim) std::copy(h->begin, h->begin+gdim, bh->begin+bdim);
    resx0 = q ? MAX(1.0, dnrm2_(&n, q->begin, &int1)) : 1.0;
    resz0 = cdim ? MAX(1.0, dnrm2_(&cdim, bh->begin, &int1)) : 1.0; 
    // initialize
    d->fill(1.0);
    kktfactor();
    matrix *x = new matrix(n, 1, 0.0);
    if(q) x->add(q->begin, -1.0);
    matrix *z = new matrix(cdim, 1);
    z->copy(bh->begin);
    matrix *u = new matrix(cdim, 1);
    kktsolver(x, z, u);
    matrix *s = new matrix(cdim, 1, 0.0);
    s->add(z->begin, -1.0);
    // ts & tz
    nrms = dnrm2_(&cdim, s->begin, &int1);
    ts = -s->min();
    tz = -z->min();
    if(ts >= -1e-8 * MAX(nrms, 1.0)){temp = ts + 1.0; s->add(temp);}
    if(tz >= -1e-8 * MAX(nrms, 1.0)){temp = tz + 1.0; z->add(temp);}
    gap = ddot_(&cdim, s->begin, &int1, z->begin, &int1);
    // steps
    matrix *rx = new matrix(n, 1);
    matrix *rz = new matrix(cdim, 1);
    matrix *dx = new matrix(*x);
    matrix *dz = new matrix(cdim, 1);
    matrix *ds = new matrix(cdim, 1);
    matrix *s2 = new matrix(cdim, 1);
    matrix *lambda = new matrix(cdim, 1);
    for(i=0; i<cdim; i++){
        d->begin[i] = std::sqrt(s->begin[i] / z->begin[i]);
        lambda->begin[i] = std::sqrt(s->begin[i] * z->begin[i]);
    }
    for(iters=0; iters<MAXITERS; iters++){
        // rx = Px + q +G'z
        rx->fill(0.0);
        dsymv_("L", &n, &dbl1, P->begin, &n, x->begin, &int1, &dbl1, rx->begin, &int1);
        f0 = 0.5 * ddot_(&n, x->begin, &int1, rx->begin, &int1);
        if(q){
            f0 += ddot_(&n, x->begin, &int1, q->begin, &int1);
            rx->add(q->begin, 1.0);
        }
        fG('T', z, rx);
        resx = dnrm2_(&n, rx->begin, &int1);
        // rz = Gx + s - h
        rz->copy(s->begin);
        rz->add(bh->begin, -1.0);
        fG('N', x, rz);
        resz = dnrm2_(&cdim, rz->begin, &int1);
        // cost
        pcost = f0;
        dcost = f0 + ddot_(&cdim, z->begin, &int1, rz->begin, &int1) - gap;
        if(pcost<0.0) relgap = - gap / pcost;
        else if(dcost>0.0) relgap = gap / dcost;
        else relgap = 1.0;
        pres = resz / resz0;
        dres = resx / resx0;
        //show_progress(iters, pcost, dcost, gap, pres, dres);
        mu = gap / cdim;
        if(pres<=FEASTOL && dres<= FEASTOL && (gap<=ABSTOL || relgap<=RELTOL)) return x;
        kktfactor();
        // f4.0
        for(i=0; i<cdim; i++) ds->begin[i] = -lambda->begin[i];
        for(i=0; i<n; i++) dx->begin[i] = -rx->begin[i];
        for(i=0; i<cdim; i++) dz->begin[i] = -rz->begin[i] - d->begin[i] * ds->begin[i];
        kktsolver(dx, dz, u);
        ds->add(dz->begin, -1.0);
        for(i=0; i<cdim; i++) s2->begin[i] = ds->begin[i] * dz->begin[i];
        dsdz = s2->sum();
        ds->divide(lambda->begin);
        dz->divide(lambda->begin);
        ts = -ds->min();
        tz = -dz->min();
        tt = MAX(0.0, MAX(ts, tz));
        step = (tt==0.0)?1.0:MIN(1.0, 1.0/tt);
        sigma = pow(MIN(1.0, MAX(0.0, 1.0-step+dsdz/gap*step*step)), EXPON);
        sigmamu = sigma * mu;
        // f4.1
        for(i=0; i<cdim; i++) ds->begin[i] = -lambda->begin[i] + (-s2->begin[i] + sigmamu) / lambda->begin[i];
        for(i=0; i<n; i++) dx->begin[i] = -rx->begin[i];
        for(i=0; i<cdim; i++) dz->begin[i] = -rz->begin[i] - d->begin[i] * ds->begin[i];
        kktsolver(dx, dz, u);
        ds->add(dz->begin, -1.0);
        dsdz = ddot_(&cdim, ds->begin, &int1, dz->begin, &int1);
        ds->divide(lambda->begin);
        dz->divide(lambda->begin);
        ts = -ds->min();
        tz = -dz->min();
        tt = MAX(0.0, MAX(ts, tz));
        step = (tt==0.0)?1.0:MIN(1.0, STEP/tt);
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
    std::cerr << "Warning: Max number of iterations reached" << std::endl;
    return x;
}



qp2::qp2(const matrix* P, const matrix* q, const matrix* lb, const matrix* rb, const matrix* G, const matrix* h):
    n(P->nrows), lbdim(lb?n:0), rbdim(rb?n:0),
    bdim(lbdim+rbdim), gdim(G?G->nrows:0),
    cdim(bdim+gdim),
    P(P), q(q), lb(lb), rb(rb), G(G), h(h),
    L(new matrix(n, n)),
    bsum(new matrix(n, 1)),
    d(new matrix(cdim, 1)),
    dsq(new matrix(cdim, 1)),
    Gd(new matrix(gdim, n)){}

void qp2::fG(char trans, const matrix* x, matrix* y) const{
    int i;
    const int int1=1;
    double s, sm;
    const double dbl1=1.0, dblm=-1.0;
    if(trans=='N'){
        s = x->sum();
        sm = -s;
        if(lbdim){
            daxpy_(&lbdim, &dblm, x->begin, &int1, y->begin, &int1);
            daxpy_(&lbdim, &s, lb->begin, &int1, y->begin, &int1);
        }
        if(rbdim){
            daxpy_(&rbdim, &dbl1, x->begin, &int1, y->begin+lbdim, &int1);
            daxpy_(&rbdim, &sm, rb->begin, &int1, y->begin+lbdim, &int1);
        }
        if(gdim) dgemv_("N", &gdim, &n, &dbl1, G->begin, &gdim, x->begin, &int1, &dbl1, y->begin+bdim, &int1);
    }else{
        s = 0.0;
        if(lbdim){
            s += ddot_(&lbdim, lb->begin, &int1, x->begin, &int1);
            for(i=0; i<n; i++) y->begin[i] -= x->begin[i];
        }
        if(rbdim){
            s -= ddot_(&rbdim, rb->begin, &int1, x->begin+lbdim, &int1);
            for(i=0; i<n; i++) y->begin[i] += x->begin[lbdim+i];
        }
        if(bdim) y->add(s);
        if(gdim) dgemv_("T", &gdim, &n, &dbl1, G->begin, &gdim, x->begin+bdim, &int1, &dbl1, y->begin, &int1);
    } 
}

void qp2::kktfactor(){
    int i, j, info, n1=n+1;
    double temp, bdb, s;
    const double dbl1=1.0;
    double *iter1;
    const double *iter01, *iter02, *iter03;
    // dsq = d{-2}
    for(i=0; i<cdim; i++) dsq->begin[i] = 1 / (d->begin[i] * d->begin[i]);
    // Gd = d{-1}G
    bdb = 0.0;
    bsum->fill(0.0);
    if(lbdim) for(i=0; i<n; i++){
        temp = lb->begin[i] * dsq->begin[i];
        bsum->begin[i] = temp;
        bdb += lb->begin[i] * temp;
    }
    if(rbdim) for(i=0; i<n; i++){
        temp = rb->begin[i] * dsq->begin[lbdim+i];
        bsum->begin[i] += temp;
        bdb += rb->begin[i] * temp;
    }
    if(gdim){
        iter01 = G->begin;
        iter1 = Gd->begin;
        iter03 = d->begin + bdim;
        for(j=0; j<n; j++){
            iter02 = iter03;
            for(i=0; i<gdim; i++) *(iter1++) = *(iter01++) / *(iter02++);
        }
    }
    // L = Gd{T}Gd + P
    for(j=0; j<n; j++){
        iter1 = L->begin+(j*n+j);
        iter01 = P->begin+(j*n+j);
        iter02 = bsum->begin+j;
        s = bdb - bsum->begin[j];
        for(i=j; i<n; i++) *(iter1++) = *(iter01++) + s - *(iter02++);
    }
    if(lbdim){
        iter1 = L->begin;
        iter01 = dsq->begin;
        for(i=0; i<n; i++, iter1+=n1, iter01++) *iter1 += *iter01;
    }
    if(rbdim){
        iter1 = L->begin;
        iter01 = dsq->begin+lbdim;
        for(i=0; i<n; i++, iter1+=n1, iter01++) *iter1 += *iter01;
    }
    if(gdim) dsyrk_("L", "T", &n, &gdim, &dbl1, Gd->begin, &gdim, &dbl1, L->begin, &n);
    dpotrf_("L", &n, L->begin, &n, &info);
}

void qp2::kktsolver(matrix* x, matrix* z, matrix* u) const{
    const int int1=1;
    z->divide(d->begin);
    u->copy(z->begin);
    z->divide(d->begin);
    fG('T', z, x);
    dtrsv_("L", "N", "N", &n, L->begin, &n, x->begin, &int1);
    dtrsv_("L", "T", "N", &n, L->begin, &n, x->begin, &int1);
    z->fill(0.0);
    fG('N', x, z);
    z->divide(d->begin);
    z->add(u->begin, -1.0);
}

matrix* qp2::solve(){
    const int int1=1;
    const int MAXITERS=100;
    const double dblm=-1.0, dbl1=1.0;
    const double EXPON=3.0, STEP=0.99, ABSTOL=1e-7, RELTOL=1e-6, FEASTOL=1e-7;
    int iters, i;
    double resx, resz, resx0, resz0, nrms;
    double f0, ts, tz, tt, temp, step;
    double pcost, dcost, pres, dres, gap, relgap;
    double sigma, mu, sigmamu, dsdz;
    // bh
    matrix *bh = new matrix(cdim, 1, 0.0);
    if(gdim) std::copy(h->begin, h->begin+gdim, bh->begin+bdim);
    resx0 = q ? MAX(1.0, dnrm2_(&n, q->begin, &int1)) : 1.0;
    resz0 = cdim ? MAX(1.0, dnrm2_(&cdim, bh->begin, &int1)) : 1.0; 
    // initialize
    d->fill(1.0);
    kktfactor();
    matrix *x = new matrix(n, 1, 0.0);
    if(q) x->add(q->begin, -1.0);
    matrix *z = new matrix(cdim, 1, 0.0);
    std::copy(h->begin, h->begin+gdim, z->begin+bdim);
    matrix *u = new matrix(cdim, 1);
    kktsolver(x, z, u);
    matrix *s = new matrix(cdim, 1, 0.0);
    s->add(z->begin, -1.0);
    // ts & tz
    nrms = dnrm2_(&cdim, s->begin, &int1);
    ts = -s->min();
    tz = -z->min();
    if(ts >= -1e-8 * MAX(nrms, 1.0)){temp = ts + 1.0; s->add(temp);}
    if(tz >= -1e-8 * MAX(nrms, 1.0)){temp = tz + 1.0; z->add(temp);}
    gap = ddot_(&cdim, s->begin, &int1, z->begin, &int1);
    // steps;
    matrix *rx = new matrix(n, 1);
    matrix *rz = new matrix(cdim, 1);
    matrix *dx = new matrix(*x);
    matrix *dz = new matrix(cdim, 1);
    matrix *ds = new matrix(cdim, 1);
    matrix *s2 = new matrix(cdim, 1);
    matrix *lambda = new matrix(cdim, 1);
    for(i=0; i<cdim; i++){
        d->begin[i] = std::sqrt(s->begin[i] / z->begin[i]);
        lambda->begin[i] = std::sqrt(s->begin[i] * z->begin[i]);
    }
    for(iters=0; iters<MAXITERS; iters++){
        // rx = Px + q +G'z
        rx->fill(0.0);
        dsymv_("L", &n, &dbl1, P->begin, &n, x->begin, &int1, &dbl1, rx->begin, &int1);
        f0 = 0.5 * ddot_(&n, x->begin, &int1, rx->begin, &int1);
        if(q){
            f0 += ddot_(&n, x->begin, &int1, q->begin, &int1);
            rx->add(q->begin, 1.0);
        }
        fG('T', z, rx);
        resx = dnrm2_(&n, rx->begin, &int1);
        // rz = Gx + s - h
        rz->copy(s->begin);
        daxpy_(&gdim, &dblm, h->begin, &int1, rz->begin+bdim, &int1);
        fG('N', x, rz);
        resz = dnrm2_(&cdim, rz->begin, &int1);
        // cost
        pcost = f0;
        dcost = f0 + ddot_(&cdim, z->begin, &int1, rz->begin, &int1) - gap;
        if(pcost<0.0) relgap = - gap / pcost;
        else if(dcost>0.0) relgap = gap / dcost;
        else relgap = 100.0;
        pres = resz / resz0;
        dres = resx / resx0;
        //show_progress(iters, pcost, dcost, gap, pres, dres);
        mu = gap / cdim;
        if(pres<=FEASTOL && dres<= FEASTOL && (gap<=ABSTOL || relgap<=RELTOL)) return x;
        kktfactor();
        // f4.0
        for(i=0; i<cdim; i++) ds->begin[i] = -lambda->begin[i];
        for(i=0; i<n; i++) dx->begin[i] = -rx->begin[i];
        for(i=0; i<cdim; i++) dz->begin[i] = -rz->begin[i] - d->begin[i] * ds->begin[i];
        kktsolver(dx, dz, u);
        ds->add(dz->begin, -1.0);
        for(i=0; i<cdim; i++) s2->begin[i] = ds->begin[i] * dz->begin[i];
        dsdz = s2->sum();
        ds->divide(lambda->begin);
        dz->divide(lambda->begin);
        ts = -ds->min();
        tz = -dz->min();
        tt = MAX(0.0, MAX(ts, tz));
        step = (tt==0.0)?1.0:MIN(1.0, 1.0/tt);
        sigma = pow(MIN(1.0, MAX(0.0, 1.0-step+dsdz/gap*step*step)), EXPON);
        sigmamu = sigma * mu;
        // f4.1
        for(i=0; i<cdim; i++) ds->begin[i] = -lambda->begin[i] + (-s2->begin[i] + sigmamu) / lambda->begin[i];
        for(i=0; i<n; i++) dx->begin[i] = -rx->begin[i];
        for(i=0; i<cdim; i++) dz->begin[i] = -rz->begin[i] - d->begin[i] * ds->begin[i];
        kktsolver(dx, dz, u);
        ds->add(dz->begin, -1.0);
        dsdz = ddot_(&cdim, ds->begin, &int1, dz->begin, &int1);
        ds->divide(lambda->begin);
        dz->divide(lambda->begin);
        ts = -ds->min();
        tz = -dz->min();
        tt = MAX(0.0, MAX(ts, tz));
        step = (tt==0.0)?1.0:MIN(1.0, STEP/tt);
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
    std::cerr << "Warning: Max number of iterations reached" << std::endl;
    return x;
}