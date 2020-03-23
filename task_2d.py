import numpy as np
import matplotlib.pyplot as plt
from md import MD

md = MD(1, 1)
md.add_molecules([[1,0,0], [0,1,0], [-1,0,0], [0,-1,0]], np.zeros((4,3)))
r, v = md.solve(1e-2, 5)
