import numpy as np
import matplotlib.pyplot as plt
from md import MD
from latice import generate_latice
from os import path

n = 4
d = 1.7
T = 165

md = MD(1, 1, d*n, True)
md.add_molecules(generate_latice(n, d), T/119.7)
r, v = md.solve(1e-2, 5)
md.velocity_autocorrelation()
msd = md.mean_squared_displacement()

D = msd[-1] / (6 * md.t[-1])

print(f"Diffusion coefficinet (msd): {D}")

plt.plot(md.t, msd)
plt.title("Mean squared displacement")
plt.show()
