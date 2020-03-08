import numpy as np
import matplotlib.pyplot as plt
from potential import LJP

p = LJP(1,1)

dt = 1e-2
T = 5
N = int(T/dt)
pN = 2

r = np.empty([N,pN,3], dtype="float64")
v = np.zeros_like(r)

r[0] = [(0,0,0), (1.5,0,0)]

track = open("track_r.xyz", "w")

for i in range(N-1):
    for j in range(pN):
        r_ = np.array(r[i], copy=True)
        r_[[0,j]] = r_[[j,0]]
        a = np.sum(p.acc(r[i,j], r_[1:], 1), axis=0)
        r[i+1,j] = r[i,j] + v[i,j] * dt
        v[i+1,j] = v[i,j] + a * dt

for i in r:
    track.write(f"{len(i)}\n\n")
    for j in i:
        track.write(f"Ar {j[0]} {j[1]} {j[2]}\n")
