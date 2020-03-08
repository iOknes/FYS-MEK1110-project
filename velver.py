import numpy as np
import matplotlib.pyplot as plt
from potential import LJP

p = LJP(1,1)

dt = 1e-2
T = 100
N = int(T/dt)
pN = 4

r = np.empty([N,pN,3], dtype="float64")
v = np.zeros_like(r)

r[0] = [(0,0,0), (1.5,0,0), (-0.25,-0.25,0), (0,0,1)]

track = open("track_r.xyz", "w")

for i in range(N-1):
    a = np.zeros((pN, 3))
    for j in range(pN):
        r_ = np.array(r[i], copy=True)
        r_[[0,j]] = r_[[j,0]]
        a[j] = np.sum(p.acc(r_[0], r_[1:], 1), axis=0)
        r[i+1,j] = r_[0] + (v[i,j] * dt) + ((a[j] / 2) * dt**2)
    for k in range(pN):
        r_ = np.array(r[i+1], copy=True)
        r_[[0,k]] = r_[[k,0]]
        a_ = np.sum(p.acc(r_[0], r_[1:], 1), axis=0)
        v[i+1,k] = v[i,k] + (a[k] + a_) / 2 * dt

for i in r:
    track.write(f"{len(i)}\n\n")
    for j in i:
        track.write(f"Ar {j[0]} {j[1]} {j[2]}\n")
