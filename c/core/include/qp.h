#pragma once

#include "matrix_base.h"
#define MIN(i, j) (((i) < (j)) ? (i) : (j))
#define MAX(i, j) (((i) > (j)) ? (i) : (j))

class qp1{
    public:
        qp1(const matrix* P, const matrix* q, const matrix* lb, const matrix* rb, const matrix* G, const matrix* h);
        matrix* solve();
    private:
        const int n, lbdim, rbdim, bdim, gdim, cdim;
        const matrix *P, *q, *lb, *rb, *G, *h;
        matrix *L, *d, *dsq, *Gd;
        void fG(char trans, const matrix* x, matrix* y) const;
        void kktfactor();
        void kktsolver(matrix* x, matrix* y, matrix* u) const;
       
};

class qp2{
    public:
        qp2(const matrix* P, const matrix* q, const matrix* lb, const matrix* rb, const matrix* G, const matrix* h);
        matrix* solve();
    private:
        const int n, lbdim, rbdim, bdim, gdim, cdim;
        const matrix *P, *q, *lb, *rb, *G, *h;
        matrix *L, *bsum, *d, *dsq, *Gd;
        void fG(char trans, const matrix* x, matrix* y) const;
        void kktfactor();
        void kktsolver(matrix* x, matrix*y, matrix* u) const;  
};