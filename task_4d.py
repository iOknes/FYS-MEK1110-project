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
r, v = md.solve(1e-2, 5, plot=True)
rdf, b_c = md.rdf(128)

plt.plot(b_c, rdf, label="Radial diffusion")
plt.legend()
plt.axhline(y=1, color="black")
plt.xlabel("Distance")
plt.ylabel("Radial distribution")
plt.savefig("rdf.png")
plt.axes([0,4,0,3])
plt.show()
