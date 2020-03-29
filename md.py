import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from potential import LJP
from time import time

class MD:
    def __init__(self, epsilon, sigma, bounding_box_size=1, p_bound=False, print_times=True):
        self.epsilon = epsilon
        self.sigma = sigma
        self.u = LJP(epsilon, sigma)
        self.r_init = np.empty(0, dtype="float64")
        self.v_init = np.empty(0, dtype="float64")
        self.pN = 0
        self.trackfile = open("track.xyz", "w")
        self.bboxlen = float(bounding_box_size)
        self.p_bound= p_bound
        self.print_times = print_times

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
    def solve(self, dt, T, plot=True):
        self.plot = plot
        p = self.u
        pN = self.pN
        N = int(T/dt)

        r = np.zeros([N,pN,3], dtype="float64")
        v = np.zeros_like(r)

        r[0] = self.r_init
        v[0] = self.v_init

        self.track(r[0])

        #Integrator loop
        s_time = time()
        for i in range(N-1):
            ai = p.acc(r[i], self.bboxlen, self.p_bound)
            r[i+1] = r[i] + (v[i] * dt) + ((ai / 2) * dt**2)
            v[i+1] = v[i] + (ai + p.acc(r[i+1], self.bboxlen, self.p_bound)) / 2 * dt

            #Try running a while to eliminate all outliars.
            if self.p_bound:
                r[i+1][r[i+1] > self.bboxlen] -= (np.floor(r[i+1][r[i+1] > self.bboxlen]/self.bboxlen) * self.bboxlen)
                r[i+1][r[i+1] < 0] -= (np.floor(r[i+1][r[i+1] < 0]/self.bboxlen) * self.bboxlen)

            self.track(r[i+1])
            print(i+1)
        
        self.r = r
        self.v = v

        if self.print_times:
            print(f"Time calculating movement: {time() - s_time:.2f}")

        self.trackfile.close()

        if plot:
            s_time = time()
            self.ek = np.sum(v**2 / 2, axis=(1,2))

            self.ep = np.zeros(N, dtype="float64")

            for i in range(N):
                for j in range(pN):
                        for k in range(j+1, pN):
                            dr = r[i,j] - r[i,k]
                            dr -= np.round(dr / self.bboxlen) * self.bboxlen
                            self.ep[i] += p(np.linalg.norm(dr))

            self.t = np.linspace(0, T, N)
            if self.print_times:
                print(f"Time calculating energies: {time() - s_time:.2f}")

            s_time = time()
            plt.plot(self.t, self.ep, label="Potential energy")
            plt.plot(self.t, self.ek, label="Kinetic energy")
            plt.plot(self.t, self.ep + self.ek, label="Total energy")
            plt.legend()
            if self.print_times:
                print(f"Time plotting: {time() - s_time:.2f}")
            plt.show()

        return r, v

    def velocity_autocorrelation(self):
        v0 = self.v[0]
        try:
            vt = np.sum(self.v * v0, axis=2)
        except NameError:
            print('System has probably not been solved. Try running the solve method!')
            exit()
        self.a = np.average(vt / np.sum(v0 * v0, axis=1), axis=1)
        if self.plot:
            plt.plot(self.t, self.a, label='Velocity autocorrelation')
            plt.show()

def md_example():
    md = MD(1, 1)
    r = [[0,0,0],[1.5,0,0]]
    v = np.zeros((2,3))
    md.add_molecules(r, v)
    r, v = md.solve(1e-2, 10, 1)

if __name__ == "__main__":
    md_example()
