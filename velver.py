import numpy as np
import matplotlib.pyplot as plt
from potential import LJP

p = LJP(1,1)

dt = 1e-2
T = 10
N = int(T/dt)
pN = 2

r = np.empty([N,pN,3], dtype="float64")
v = np.zeros_like(r)

r[0] = [(0,0,0), (1.5,0,0)]
#r[0] = [(0,0,0), (1.5,0,0), (-0.25,-0.25,0), (0,0,1)]

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

#Calculate energies for all particles
#Kinetic energy
ek = np.sum(np.linalg.norm(v, axis=2)**2 / 2, axis=1)
#Potential energy
d = np.empty((N, pN, pN-1, 3), dtype="float64")
print(np.shape(d))
for i in range(N):
    for j in range(pN):
        r_ = np.array(r[i], copy=True)
        r_[[0,j]] = r_[[j,0]]
        d[i,j] = r_[0] - r_[1:]
ep = np.sum(np.sum(p(np.linalg.norm(d, axis=3)), axis=2), axis=1)

#Generate time array and plot energies
t = np.linspace(0, T, N)
plt.plot(t, ek, label="Kinetic energy")
plt.plot(t, ep, label="Potential energy")
plt.plot(t, ep + ek, label="Total energy")
plt.legend()
plt.show()
