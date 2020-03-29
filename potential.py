import numpy as np

def norm(x):
    lambda x: np.sum(np.sqrt(x**2))

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

    #Returns the acceleration array for any given array of positions
    def acc(self, x, L=1, pbound=False):
        a = np.zeros((len(x), len(x),3))
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                dr = x[i] - x[j]
                if pbound:
                    dr = dr - np.round(dr/L)*L
                r = np.linalg.norm(dr)
                if r < 3*self.sigma:
                    a[i,j] = -(24*(2*(r)**(-12) - (r)**(-6)) * (dr) / (r)**2)
                    a[j,i] = -a[i,j]
                else:
                    a[i,j] = 0
                    a[j,i] = 0
        return np.sum(a,axis = 0)
