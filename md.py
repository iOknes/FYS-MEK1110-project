import numpy as np
from potential import LJP

class MD:
    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma
        self.u = LJP(epsilon, sigma)
        self.r = np.empty(0)
        self.v = np.empty(0)

    #Takes one position vector and one velocity vector of any equal dimension
    def add_molecule(self, position, velocity):
        np.concatenate(self.r, np.array(position))
        np.concatenate(self.v, np.array(velocity))
    
    def add_molecules(self, positions, velocities):
        for i in range(len(positions)):
            self.add_molecule(positions[i], velocities[i])
    
    

