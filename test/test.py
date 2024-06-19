import numpy as np

def read_csv(path):
    #A = pd.read_csv(path, sep=",", header=None).values
    A = np.loadtxt(path, delimiter=",", dtype=np.double)
    return A

P = read_csv("../../data/P.csv")
q = read_csv("../../data/q.csv")
G0 = read_csv("../../data/G0.csv").reshape(1, -1)
h0 = read_csv("../../data/h0.csv").reshape(-1)
G1 = read_csv("../../data/G1.csv")
h1 = read_csv("../../data/h1.csv")
G2 = read_csv("../../data/G2.csv")
h2 = read_csv("../../data/h2.csv")
lb = read_csv("../../data/lb.csv")
rb = read_csv("../../data/rb.csv")


import cudacore
cuP = cudacore.matrix_fromBuffer(P)
culb = cudacore.matrix_fromBuffer(lb)
curb = cudacore.matrix_fromBuffer(rb)
cuG0 = cudacore.matrix_fromBuffer(G0)
cuh0 = cudacore.matrix_fromBuffer(h0)
cublas_handle = cudacore.CublasHandle()
cusolver_handle = cudacore.CusolverHandle()
solver = cudacore.cuqp1(cublas_handle, cusolver_handle, P=cuP, lb=culb, rb=curb, G=cuG0, h=cuh0)
x = solver.solve()
print(np.array(x).squeeze()[:5])