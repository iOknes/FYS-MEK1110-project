import numpy as np
import matplotlib.pyplot as plt
from md import MD
from latice import generate_latice
from os import path

n = 4
d = 1.7
T = 94.4

md = MD(1, 1, d*n, True)
md.add_molecules(generate_latice(n, d), T/119.7)
r, v = md.solve(1e-2, 5, True)

"""
rdf = md.rdf(4*n**3)
t = np.linspace(0, T, int(5/1e-2))

plt.plot(t, rdf, label="Radial diffusion")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Radial distribution")
plt.savefig("rdf.png")
plt.show()
"""