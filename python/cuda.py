import numpy as np
import qptools.cudacore as cudacore

class Handle:
    def __init__(self):
        self.cublas_handle = cudacore.CublasHandle()
        self.cusolver_handle = cudacore.CusolverHandle()

def matrix(A):
    if isinstance(A, cudacore.matrix):
        return A
    elif isinstance(A, np.ndarray):
        if A.ndim == 0:
            return cudacore.matrix_fromBuffer(A.reshape(-1))
        elif A.ndim in {1, 2}:
            if A.dtype != np.double: A = A.asdtype(np.double)
            return cudacore.matrix_fromBuffer(A)
        else:
            raise ValueError("The dimension of input should be 1 or 2, but it is {A.ndim}")
    else:
        raise ValueError(f"The input should be np.ndarray, but it is {A.__class__}")

def qp1(handle, P, q=None, lb=None, rb=None, G=None, h=None):
    P = matrix(P)
    n = P.nrows
    if P.ncols != n:
        raise ValueError("P should be a square matrix")
    if q is not None:
        q = matrix(q)
        if q.size != n:
            raise ValueError(f"q should be an array of size {n}")
    if G is not None:
        G = matrix(G)
        g = G.nrows
        if G.ncols != n:
            raise ValueError(f"G should be a matrix with {n} columns")
        h = matrix(h)
        if h.size != g:
            raise ValueError(f"h should be an array of size {g}")
    if lb is not None:
        lb = matrix(lb)
        if lb.size != n:
            raise ValueError(f"lb should be an array of size {n}")
    if rb is not None:
        rb = matrix(rb)
        if rb.size != n:
            raise ValueError(f"rb should be an array of size {n}")
    solver = cudacore.qp1(handle.cublas_handle, handle.cusolver_handle, P=P, q=q, lb=lb, rb=rb, G=G, h=h)
    x = solver.solve()
    return np.array(x).squeeze()


def qp2(handle, P, q=None, lb=None, rb=None, G=None, h=None):
    P = matrix(P)
    n = P.nrows
    if P.ncols != n:
        raise ValueError("P should be a square matrix")
    if q is not None:
        q = matrix(q)
        if q.size != n:
            raise ValueError(f"q should be an array of size {n}")
    if G is not None:
        G = matrix(G)
        g = G.nrows
        if G.ncols != n:
            raise ValueError(f"G should be a matrix with {n} columns")
        h = matrix(h)
        if h.size != g:
            raise ValueError(f"h should be an array of size {g}")
    if lb is not None:
        lb = matrix(lb)
        if lb.size != n:
            raise ValueError(f"lb should be an array of size {n}")
    if rb is not None:
        rb = matrix(rb)
        if rb.size != n:
            raise ValueError(f"rb should be an array of size {n}")
    solver = cudacore.qp2(handle.cublas_handle, handle.cusolver_handle, P=P, q=q, lb=lb, rb=rb, G=G, h=h)
    x = solver.solve()
    return np.array(x).squeeze()
