import numpy as np

norm = lambda x: np.sum(np.sqrt(x**2))

class LJP:
    def __init__(self, eps, sgm):
        self.epsilon = eps
        self.sigma = sgm
    """
    Return the potential between two atoms at a distance r sigma
    Takes either int, float or array as argument. Otherwise raises TypeError
    """
    def __call__(self, r):
        if type(r) is np.ndarray:
            r_ = r > 3
            r = 4 * (((r)**(-12) - (r)**(-6)) - ((3)**(-12) - (3)**(-6)))
            r[r_] = 0
            return r
        else:
            if r < 3:
                return 4 * (((r)**(-12) - (r)**(-6)) - ((3)**(-12) - (3)**(-6)))
            else:
                return 0

    #Returns the acceleration excerten on r1 by r2
    def acc(self, r1, r2):
        r = r1 - r2
        rn = norm(r)
        if rn < 3:
            return 24 * (2 * (rn)**(-12) - (rn)**(-6)) * r / (rn**2)
            #return 24 * self.epsilon * r / rn**2 * ((2 * (self.sigma / rn)**12 - (self.sigma / rn)**6) - ((self.sigma / rn)**12 - (self.sigma / rn)**6))
        else:
            return 0
