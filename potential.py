import numpy as np

def norm(x):
    lambda x: np.sum(np.sqrt(x**2))

class LJP:
    def __init__(self, eps, sgm, rc):
        self.epsilon = eps
        self.sigma = sgm
        self.rc = rc
        self.ljrc = 4 *((1/rc)**(12) - (1/rc)**(6))
    """
    Return the potential between two atoms at a distance r sigma
    Takes either int, float or array as argument. Otherwise raises TypeError
    """

    #Returns the acceleration array for any given array of positions
    def __call__(self, x, L=1, pbound=False):
        a = np.zeros((len(x), len(x),3))
        p = 0
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                dr = x[i] - x[j]
                if pbound:
                    dr = dr - np.round(dr/L)*L
                r = np.linalg.norm(dr)
                p += 4 *((1/r)**(12) - (1/r)**(6)) - self.ljrc
                if r < self.rc:
                    a[i,j] = -(24*(2*(r)**(-12) - (r)**(-6)) * (dr) / (r)**2)
                    a[j,i] = -a[i,j]
                else:
                    a[i,j] = 0
                    a[j,i] = 0
        return np.sum(a,axis = 0), p
