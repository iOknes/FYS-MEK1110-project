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
        if type(r) is float or type(r) is int:
            if r < 3:
                return 4 * self.epsilon * (((self.sigma / r)**12 - (self.sigma / r)**6) - ((self.sigma / 3)**12 - (self.sigma / 3)**6))
            else:
                return 0
        elif type(r) is np.ndarray:
            print(r)
            r_ = r > 3
            print(r_)
            r = 4 * self.epsilon * (((self.sigma / r)**12 - (self.sigma / r)**6) - ((self.sigma / 3)**12 - (self.sigma / 3)**6))
            print(r)
            r[r_] = 0
            print(r)
            return r
        else:
            raise TypeError("Unsupported argument type!")

    #Returns the acceleration excerten on r1 by r2
    def acc(self, r1, r2, m1):
        r = r1 - r2
        rn = norm(r)
        if rn < 3:
            return 24 * self.epsilon / m1 * r / rn**2 * (2 * ((self.sigma / rn)**12 - (self.sigma / rn)**6) - ((self.sigma / rn)**12 - (self.sigma / rn)**6))
        else:
            return 0
