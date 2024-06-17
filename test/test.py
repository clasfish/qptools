import numpy as np
import pandas as pd
import cvxopt

def get_matrix(path):
    A = pd.read_csv(path, sep=",", header=None).values
    return cvxopt.matrix(A)
P = get_matrix("/home/longxin/qpdata2/P.csv")
q = get_matrix("/home/longxin/qpdata2/q.csv")
G = get_matrix("/home/longxin/qpdata2/G2.csv")
h = get_matrix("/home/longxin/qpdata2/h2.csv")
result = cvxopt.solvers.qp(P, q, G, h)
if result is not None:
    print(list(result)[:5])