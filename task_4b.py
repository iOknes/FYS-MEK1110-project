import numpy as np
from md import MD
from latice import generate_latice
from os import path

n = 4
d = 1.7
T = 94.4

md = MD(1, 1, d*n, False)
if path.exists("cache/r_last.txt") and path.exists("cache/v_last.txt"):
    try:
        md.add_molecules(np.loadtxt("cache/r_last.txt"), np.loadtxt("cache/v_last.txt"))
    except IndexError:
        md.add_molecules(generate_latice(n, d), np.random.normal(0, np.sqrt(T/119.7), size=(4*n**3, 3)))
else:
    md.add_molecules(generate_latice(n, d), np.random.normal(0, np.sqrt(T/119.7), size=(4*n**3, 3)))
r, v = md.solve(1e-2, 5)
md.velocity_autocorrelation()
np.savetxt("cache/r_last.txt", r[-1])
np.savetxt("cache/v_last.txt", v[-1])
