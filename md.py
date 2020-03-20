import numpy as np
from potential import LJP

class MD:
    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma
        self.u = LJP(epsilon, sigma)
        self.r_init = np.empty(0, dtype="float64")
        self.v_init = np.empty(0, dtype="float64")
        self.pN = 0
        self.trackfile = open("track_r.xyz", "w")

    #Takes one position vector and one velocity vector of any equal dimension
    def add_molecule(self, position, velocity):
        self.pN += 1
        self.r_init = np.split(np.concatenate((self.r_init, np.array(position)), axis=None), self.pN)
        self.v_init = np.split(np.concatenate((self.v_init, np.array(velocity)), axis=None), self.pN)

    def add_molecules(self, positions, velocities):
        for i in range(len(positions)):
            self.add_molecule(positions[i], velocities[i])
    
    def track(self, r):
        self.trackfile.write(f"{len(r)}\ntype x y z\n")
        for i in r:
            self.trackfile.write(f"Ar {i[0]} {i[1]} {i[2]}\n")

    """
    Runs the simulation for a given time step, time and mass using the Velocity Verlet method
    Arguments:
    dt: Time step
    T: End time
    m: Mass of particles
    """
    def solve(self, dt, T, m):
        p = self.u
        pN = self.pN
        N = int(T/dt)

        r = np.empty([N,pN,3], dtype="float64")
        v = np.zeros_like(r)

        r[0] = self.r_init
        v[0] = self.v_init

        self.track(r[0])

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

            self.track(r[i+1])

        return r, v

def md_example():
    md = MD(1, 1)
    r = [[0,0,0],[1.5,0,0]]
    v = np.zeros((2,3))
    md.add_molecules(r, v)
    r, v = md.solve(1e-2, 10, 1)

if __name__ == "__main__":
    md_example()
