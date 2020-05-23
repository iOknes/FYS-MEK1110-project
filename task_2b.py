import numpy as np
from md import MD

md = MD(1,1)
md.add_molecules([(0,0,0),(1.5,0,0)], np.zeros([2,3]))
md.solve(1e-2, 5)
