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

r[0] = [(0,0,0),(1.5,0,0)]
#r[0] = [(0,0,0), (1.5,0,0), (-1.5,0,0)]
#r[0] = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)]

track = open("track_r.xyz", "w")

#Does the numerical integration using prebaking of acceleration
for i in range(N-1):
    a = np.zeros((pN, 3))
    a_ = np.zeros((pN, 3))
    for j in range(pN):
        for k in range(j+1, pN):
            a__ = p.acc(r[i,j], r[i,k])
            a[j] += a__
            a[k] -= a__
    for j in range(pN):
        r[i+1,j] = r[i,j] + (v[i,j] * dt) + ((a[j] / 2) * dt**2)
    for j in range(pN):
        for k in range(j+1, pN):
            a__ = p.acc(r[i+1,j], r[i+1,k])
            a[j] += a__
            a[k] -= a__
    for j in range(pN):
        v[i+1,j] = v[i,j] + (a[j] + a_[j]) / 2 * dt

for i in r:
    track.write(f"{len(i)}\ntype x y z\n")
    for j in i:
        track.write(f"Ar {j[0]} {j[1]} {j[2]}\n")

#Calculate energies for all particles
#Kinetic energy
#ek = np.sum(pN * m * np.linalg.norm(v, axis=2)**2 / 2, axis=1)
ek = np.sum(v**2 / 2, axis=(1,2))
#Potential energy
"""d = np.empty((N, pN, pN-1, 3), dtype="float64")
for i in range(N):
    for j in range(pN):
        r_ = np.array(r[i], copy=True)
        r_[[0,j]] = r_[[j,0]]
        d[i,j] = r_[0] - r_[1:]
ep = np.sum(np.sum(p(np.linalg.norm(d, axis=3)), axis=2), axis=1)"""

ep = np.zeros([N, pN], dtype="float64")

for i in range(N):
    for j in range(pN):
            for k in range(j+1, pN):
                ep_ = p(np.linalg.norm(r[i,j] - r[i,k]))
                ep[i,j] = ep_

ep = np.sum(ep, axis=1)

#Generate time array and plot energies
t = np.linspace(0, T, N)
plt.plot(t, ek, label="Kinetic energy")
plt.plot(t, ep, label="Potential energy")
plt.plot(t, ep + ek, label="Total energy")
plt.legend()
plt.show()
