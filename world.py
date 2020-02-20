"""
A storage class for bodies, containing the workings of natural laws
"""
import numpy as np
from body import Body
from rigidbody import RigidBody

class World:
    def __init__(self, eps=1, sgm=1):
        self.eps = eps
        self.sgm = sgm
        self.bodies = np.empty(0, dtype=Body)

    def joinBody(self, body):
        self.bodies = np.concatenate((self.bodies, body), axis=None)

    def addBody(self, pos, vel, mass, charge, rigid=False):
        body = RigidBody(pos, mass, charge) if rigid else Body(pos, vel, mass, charge)
        self.joinBody(body)

    @property
    def dim(self):
        try:
            return len(self.bodies[0].pos)
        except TypeError:
            print("No bodies in world!")

    def getAcc(self):
        n = len(self.bodies)
        acc = np.empty((n, self.dim))
        for i in range(n):
            bodies = self.bodies[self.bodies != self.bodies[i]]
            r = Body.getPos(bodies)
            r -= self.bodies[i].pos
            rn = np.linalg.norm(r, 1)
            ar = 24 * (2 * rn**-12 - rn**-6) * np.transpose(r)/rn**2
            acc[i] = np.sum(ar, 1)
        return -acc

    def __mul__(self, dt):
        a = self.getAcc()
        self.bodies = Body.move(self.bodies, a, dt)
        return self

    def __imul__(self, dt):
        return self * dt
