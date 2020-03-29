import numpy as np
from md import MD

r = [[0.4,0,0], [1.6,0,0]]
v = [[0,0,0], [0,0,0]]

md = MD(1, 1, 3, True)
md.add_molecules(r, v)
md.solve(1e-2, 5, True)
md.velocity_autocorrelation()
