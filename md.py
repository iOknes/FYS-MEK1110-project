import numpy as np
import matplotlib.pyplot as plt
from potential import LJP
from time import time
import os

class MD:
    def __init__(self, epsilon, sigma, bounding_box_size=1, p_bound=True, print_times=True, rc=3):
        self.epsilon = epsilon
        self.sigma = sigma
        self.u = LJP(epsilon, sigma, rc)
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

    """
    positions: array of position of shape (N x Dim)
    velocity: array of velocities of shape (N x Dim) or a scaled temprature for generating thermal velocities
    """
    def add_molecules(self, positions, velocities):
        if type(velocities) is float:
            self.temp = velocities
            self.add_molecules(positions, np.random.normal(0, np.sqrt(velocities), size=positions.shape))
        else:
            if not hasattr(self, 'temp'):
                self.temp = np.math.nan
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
    def solve(self, dt, T, energy=True, plot=True, cache=True):
        self.plot = plot
        p = self.u
        pN = self.pN
        N = int(T/dt)

        loaded = False
        created = False
        
        self.wallpass = np.zeros((N, pN, 3), dtype="float64")

        r = np.zeros([N,pN,3], dtype="float64")
        v = np.zeros_like(r)
        self.ep = np.zeros(N, dtype="float64")

        s_time = time()

        if cache:
            fN = 0
            while not (loaded or created):
                if not os.path.exists(f"cache/{fN}"):
                    os.mkdir(f"cache/{fN}")
                    created = True
                elif np.load(f"cache/{fN}/check.npy", allow_pickle=True).item() == {"dt": dt, "T": T, "epsilon": self.epsilon, "sigma": self.sigma, "bboxlen": self.bboxlen, "p_bound": self.p_bound, "temp": self.temp} and np.all(np.load(f"cache/{fN}/r_check.npy") == self.r_init):
                    print(f"Loading cache file {fN}...")
                    r = np.load(f"cache/{fN}/r.npy")
                    v = np.load(f"cache/{fN}/v.npy")
                    self.ep = np.load(f"cache/{fN}/ep.npy")
                    #self.wallpass = np.load(f"cache/{fN}/wallpass.npy")
                    loaded = True
                else:
                    fN += 1
        else:
            created = True

        if created:
            r[0] = self.r_init
            v[0] = self.v_init

            self.track(r[0])

            #Integrator loop
            for i in range(N-1):
                ai, self.ep[i] = p(r[i], self.bboxlen, self.p_bound)
                r[i+1] = r[i] + (v[i] * dt) + ((ai / 2) * dt**2)
                aii, self.ep[i+1] = p(r[i+1], self.bboxlen, self.p_bound)
                v[i+1] = v[i] + (ai + aii) / 2 * dt

                #Try running a while to eliminate all outliars.
                if self.p_bound:
                    self.wallpass[i+1][r[i+1] > self.bboxlen] = self.wallpass[i][r[i+1] > self.bboxlen] + 1
                    self.wallpass[i+1][r[i+1] < 0] = self.wallpass[i][r[i+1] < 0] - 1
                    r[i+1][r[i+1] > self.bboxlen] -= (np.floor(r[i+1][r[i+1] > self.bboxlen]/self.bboxlen) * self.bboxlen)
                    r[i+1][r[i+1] < 0] -= (np.floor(r[i+1][r[i+1] < 0]/self.bboxlen) * self.bboxlen)

                self.track(r[i+1])
                print(i+1)
            
        self.r = r
        self.v = v

        self.ek = np.sum(v**2 / 2, axis=(1,2))
        self.t = np.linspace(0, T, N)
        
        if cache and not loaded and created:
            np.save(f"cache/{fN}/r.npy", r)
            np.save(f"cache/{fN}/v.npy", v)
            np.save(f"cache/{fN}/ep.npy", self.ep)
            np.save(f"cache/{fN}/wallpass.npy", self.wallpass)
            np.save(f"cache/{fN}/check.npy", {"dt": dt, "T": T, "epsilon": self.epsilon, "sigma": self.sigma, "bboxlen": self.bboxlen, "p_bound": self.p_bound, "temp": self.temp})
            np.save(f"cache/{fN}/r_check.npy", self.r_init)

        if self.print_times:
            print(f"Time calculating movement: {time() - s_time:.2f}")

        self.trackfile.close()

        if plot:

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
        self.a = np.sum(vt / np.sum(v0 * v0, axis=1), axis=1) / self.pN
        if self.plot:
            plt.plot(self.t, self.a, label='Velocity autocorrelation')
            plt.xlabel("Time")
            plt.ylabel("Velocity autocorrelation")
            plt.show()
        D = np.trapz(self.a, self.t)/3
        print(f"Diffuion constant: {D}")
        return D

    def mean_squared_displacement(self, start_t):
        msd = np.sum((self.r[start_t:] - self.r[start_t] + self.wallpass[start_t:] * self.bboxlen)**2, axis=(1,2))/self.pN
        return msd

    def rdf(self, bin_num):
        """
        bin_edges = edges of bins. Typically np.linspace(0, rc, num_bins+1) for some cut-off rc.
        r = Nx3-array of positions of atoms at a given timestep.
        V = volume of system.
        """

        for r in self.r[20:]:

            bin_edges = np.linspace(0, self.u.rc, bin_num)
            V = self.bboxlen**3

            rdf = []

            N = r.shape[0]
            bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_sizes = bin_edges[1:] - bin_edges[:-1]
            n = np.zeros_like(bin_sizes)
            for i in range(N):
                dr = np.linalg.norm(r - r[i], axis=1)       # Distances from atom i.
                n += np.histogram(dr, bins=bin_edges)[0]    # Count atoms within each
                                                            # distance interval.
            n[0] = 0
            # Equation (7) on the preceding page:
            #rdf = V / N**2 * n / (4 * np.pi * bin_centres**2 * bin_sizes)
            rdf.append(V / N**2 * n / (4 * np.pi * bin_centres**2 * bin_sizes))

        #return np.average(rdf, axis=0), bin_centres
        return rdf[-1], bin_centres

def md_example():
    md = MD(1, 1)
    r = [[0,0,0],[1.5,0,0]]
    v = np.zeros((2,3))
    md.add_molecules(r, v)
    r, v = md.solve(1e-2, 10, 1)

if __name__ == "__main__":
    md_example()
