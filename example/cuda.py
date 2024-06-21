import numpy as np
import pandas as pd
import qptools.cuda
handle = qptools.cuda.Handle()

def read_csv(path):
    A = pd.read_csv(path, sep=",", header=None).values
    return A

P = read_csv("../../data/P.csv")
q = read_csv("../../data/q.csv")
G0 = read_csv("../../data/G0.csv")
h0 = read_csv("../../data/h0.csv")
G1 = read_csv("../../data/G1.csv")
h1 = read_csv("../../data/h1.csv")
G2 = read_csv("../../data/G2.csv")
h2 = read_csv("../../data/h2.csv")
lb = read_csv("../../data/lb.csv")
rb = read_csv("../../data/rb.csv")

x = qptools.cuda.qp1(handle, P=P, G=G1, h=h1)
print(x)