import numpy as np
from ljp import LJP

norm = lambda x: np.sqrt(np.sum(x**2))

u = LJP(1, 1)

#a_ij = lambda r_i, r_j: 24 * (2 * norm(r_i - r_j)**-12 - norm(r_i - r_j)**-6) * norm(r_i - r_j)/norm(r_i - r_j)**2
a_ij = lambda i, j: i * j

def a(p):
    acc = np.empty((len(p), 3))
    for i in range(len(p)):
        j = p[p != p[i]]
        try:
            acc[i] = np.sum(np.transpose(a_ij(i, j)), 1)
        except IndexError:
            print("Possible duplication of coordinates")
    return acc


