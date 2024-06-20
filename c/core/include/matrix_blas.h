#pragma once

extern "C"{
    // nrm2, scal
    double dnrm2_(const int *n, const double *x, const int *incx);
    float snrm2_(const int *n, const float *x, const int *incx);
    void dscal_(const int *n, const double *alpha, double *x, const int *incx);
    void sscal_(const int *n, const float *alpha, float *x, const int *incx);
    // dot, axpy
    double ddot_(const int *n, const double *x, const int *incx, const double *y, const int *incy);
    float sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy);
    void daxpy_(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);
    void saxpy_(const int *n, const float *alpha, const float *x, const int *incx, float *y, const int *incy);
    // trmv, trsv, symv
    void dtrmv_(const char *uplo, const char *trans, const char *diag, const int *n, const double *A, const int *lda, double *x, const int *incx);
    void strmv_(const char *uplo, const char *trans, const char *diag, const int *n, const float *A, const int *lda, float *x, const int *incx);
    void dtrsv_(const char *uplo, const char *trans, const char *diag, const int *n, const double *A, const int *lda, double *x, const int *incx);
    void strsv_(const char *uplo, const char *trans, const char *diag, const int *n, const float *A, const int *lda, float *x, const int *incx);
    void dgemv_(const char *trans, const int *m, const int *n, const double *alpha, const double *A, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy);
    void sgemv_(const char *trans, const int *m, const int *n, const float *alpha, const float *A, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy);
    void dsymv_(const char *uplo, const int *n, const double *alpha, const double *A, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy);
    void ssymv_(const char *uplo, const int *n, const float *alpha, const float *A, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy);
    // syrk
    void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k, const double *alpha, const double *A, const int *lda, const double *beta, double *B, const int *ldb);
    void ssyrk_(const char *uplo, const char *trans, const int *n, const int *k, const float *alpha, const float *A, const int *lda, const float *beta, float *B, const int *ldb);
    // potrf
    void dpotrf_(const char *uplo, const int *n, double *A, const int *lda, int *info);
    void spotrf_(const char *uplo, const int *n, float *A, const int *lda, int *info);
}

/*
inline double min_(const int *n, const double *x, const int *incx){
    return dmin_(n, x, incx);
}
inline float min_(const int *n, const float *x, const int *incx){
    return smin_(n, x, incx);
}
inline double max_(const int *n, const double *x, const int *incx){
    return dmax_(n, x, incx);
}
inline float max_(const int *n, const float *x, const int *incx){
    return smax_(n, x, incx);
}

inline double sum_(const int *n, const double *x, const int *incx){
    return dsum_(n, x, incx);
}
inline float sum_(const int *n, const float *x, const int *incx){
    return ssum_(n, x, incx);
}

inline double nrm2_(const int *n, const double *x, const int *incx){
    return dnrm2_(n, x, incx);
}
inline float nrm2_(const int *n, const float *x, const int *incx){
    return snrm2_(n, x, incx);
}
inline void scal_(const int *n, const double *alpha, double *x, const int *incx){
    dscal_(n, alpha, x, incx);
}
inline void scal_(const int *n, const float *alpha, float *x, const int *incx){
    sscal_(n, alpha, x, incx);
}
//-----------------------------------------------------------------
inline double dot_(const int *n, const double *x, const int *incx, const double *y, const int *incy){
    return ddot_(n, x, incx, y, incy);
}
inline float dot_(const int *n, const float *x, const int *incx, const float *y, const int *incy){
    return sdot_(n, x, incx, y, incy);
}
inline void axpy_(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy){
    daxpy_(n, alpha, x, incx, y, incy);
}
inline void axpy_(const int *n, const float *alpha, const float *x, const int *incx, float *y, const int *incy){
    saxpy_(n, alpha, x, incx, y, incy);
}
//-----------------------------------------------------------------
inline void trmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const double *A,
    const int *lda,
    double *x,
    const int *incx
){
    dtrmv_(uplo, trans, diag, n, A, lda, x, incx);
}

inline void trmv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const float *A,
    const int *lda,
    float *x,
    const int *incx
){
    strmv_(uplo, trans, diag, n, A, lda, x, incx);
}

inline void trsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const double *A,
    const int *lda,
    double *x,
    const int *incx
){
    dtrsv_(uplo, trans, diag, n, A, lda, x, incx);
}

inline void trsv_(
    const char *uplo,
    const char *trans,
    const char *diag,
    const int *n,
    const float *A,
    const int *lda,
    float *x,
    const int *incx
){
    strsv_(uplo, trans, diag, n, A, lda, x, incx);
}

void gemv_(
    const char *trans,
    const int *m,
    const int *n,
    const double *alpha,
    const double *A,
    const int *lda,
    const double *x,
    const int *incx,
    const double *beta,
    double *y,
    const int *incy
){
    dgemv_(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void gemv_(
    const char *trans,
    const int *m,
    const int *n,
    const float *alpha,
    const float *A,
    const int *lda,
    const float *x,
    const int *incx,
    const float *beta,
    float *y,
    const int *incy
){
    sgemv_(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void symv_(
    const char *uplo, 
    const int *n,
    const double *alpha,
    const double *A,
    const int *lda,
    const double *x,
    const int *incx,
    const double *beta,
    double *y,
    const int *incy
){
    dsymv_(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

void symv_(
    const char *uplo, 
    const int *n,
    const float *alpha,
    const float *A,
    const int *lda,
    const float *x,
    const int *incx,
    const float *beta,
    float *y,
    const int *incy
){
    ssymv_(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}
//-----------------------------------------------------------------
void syrk_(
    const char *uplo,
    const char *trans,
    const int *n,
    const int *k,
    const double *alpha,
    const double *A,
    const int *lda,
    const double *beta,
    double *B,
    const int *ldb
){
    dsyrk_(uplo, trans, n, k, alpha, A, lda, beta, B, ldb);
}

void syrk_(
    const char *uplo,
    const char *trans,
    const int *n,
    const int *k,
    const float *alpha,
    const float *A,
    const int *lda,
    const float *beta,
    float *B,
    const int *ldb
){
    ssyrk_(uplo, trans, n, k, alpha, A, lda, beta, B, ldb);
}
//-----------------------------------------------------------------
void potrf_(
    const char *uplo, 
    const int *n,
    double *A,
    const int *lda,
    int *info
){
    dpotrf_(uplo, n, A, lda, info);
}

void potrf_(
    const char *uplo, 
    const int *n,
    float *A,
    const int *lda,
    int *info
){
    spotrf_(uplo, n, A, lda, info);
}
*/