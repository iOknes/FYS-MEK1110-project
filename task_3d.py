import numpy as np
from md import MD
from latice import generate_latice

n = 5
d = 1.7
T = 5

md = MD(1, 1, d*n, True)
md.add_molecules(generate_latice(n, d), np.random.normal(0, np.sqrt(T), size=(4*n**3, 3)))
r, v = md.solve(1e-2, 5)
md.velocity_autocorrelation()
