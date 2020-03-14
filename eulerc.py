import numpy as np
import matplotlib.pyplot as plt
from potential import LJP

p = LJP(1,1)
m = 1

dt = 1e-2
T = 10
N = int(T/dt)
pN = 2

r = np.empty([N,pN,3], dtype="float64")
v = np.zeros_like(r)

r[0] = [(0,0,0), (1.5,0,0)]

for i in range(N-1):
    for j in range(pN):
        r_ = np.array(r[i], copy=True)
        r_[[0,j]] = r_[[j,0]]
        a = np.sum(p.acc(r[i,j], r_[1:], m), axis=0)
        v[i+1,j] = v[i,j] + a * dt
        r[i+1,j] = r[i,j] + v[i+1,j] * dt

track = open("track_r.xyz", "w")

for i in r:
    track.write(f"{len(i)}\n\n")
    for j in i:
        track.write(f"Ar {j[0]} {j[1]} {j[2]}\n")

track.close()

#Generate time array
t = np.linspace(0, T, N)

#Plot distance between two atoms
d = np.linalg.norm(r[:,0] - r[:,1], axis=1)
plt.plot(t, d, label="Distance between particles 0 and 1")
plt.title("Distance between atoms against time")
plt.xlabel("Time")
plt.ylabel("Distance (Sigma)")
plt.legend()
plt.grid()
plt.show()

plt.cla()

#Calculate energies for all particles
#Kinetic energy
ek = np.sum(pN * m * np.linalg.norm(v, axis=2)**2 / 2, axis=1)
#Potential energy
d = np.empty((N, pN, pN-1, 3), dtype="float64")
print(np.shape(d))
for i in range(N):
    for j in range(pN):
        r_ = np.array(r[i], copy=True)
        r_[[0,j]] = r_[[j,0]]
        d[i,j] = r_[0] - r_[1:]
ep = np.sum(np.sum(p(np.linalg.norm(d, axis=3)), axis=2), axis=1)

#Plot energies
plt.plot(t, ek, label="Kinetic energy")
plt.plot(t, ep, label="Potential energy")
plt.plot(t, ep + ek, label="Total energy")
plt.title("Kinetic, potential and total energy at 0.95 sigma separation")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.show()
