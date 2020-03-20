import numpy as np
from potential import LJP

class MD:
    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma
        self.u = LJP(epsilon, sigma)
        self.r = np.empty(0)
        self.v = np.empty(0)
        self.pN = 0

    #Takes one position vector and one velocity vector of any equal dimension
    def add_molecule(self, position, velocity):
        np.concatenate(self.r, np.array(position))
        np.concatenate(self.v, np.array(velocity))
        self.pN += 1

    def add_molecules(self, positions, velocities):
        for i in range(len(positions)):
            self.add_molecule(positions[i], velocities[i])
    
    def solve(self, dt, T, m):
        p = self.u
        pN = self.pN
        N = int(T/dt)

        r = np.empty([N,pN,3], dtype="float64")
        v = np.zeros_like(r)

        r[0] = self.r
        v[0] = self.v

        track = open("track_r.txt", "w")

        for i in range(N-1):
            a = np.zeros((pN, 3))
            a_ = np.zeros((pN, 3))
            for j in range(pN):
                for k in range(j+1, pN):
                    a__ = p.acc(r[i,j], r[i,k], m)
                    a[j] += a__
                    a[k] -= a__
            for j in range(pN):
                r[i+1,j] = r[i,j] + (v[i,j] * dt) + ((a[j] / 2) * dt**2)
            for j in range(pN):
                for k in range(j+1, pN):
                    a__ = p.acc(r[i+1,j], r[i+1,k], m)
                    a[j] += a__
                    a[k] -= a__
            for j in range(pN):
                v[i+1,j] = v[i,j] + (a[j] + a_[j]) / 2 * dt

        for i in r:
            track.write(f"{len(i)}\ntype x y z\n")
            for j in i:
                track.write(f"Ar {j[0]} {j[1]} {j[2]}\n")
