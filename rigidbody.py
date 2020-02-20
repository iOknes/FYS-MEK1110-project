"""
A subclass of body for simulating rigid bodies
"""
from body import Body

class RigidBody(Body):
    def __init__(self, pos, mass, charge):
        super().__init__(pos, 0, mass, charge)

    @staticmethod
    def fromBody(body):
        return RigidBody(body.pos, body.mass, body.charge)
