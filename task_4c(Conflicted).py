import numpy as np
import matplotlib.pyplot as plt
from md import MD
from latice import generate_latice
import os

dt = 1e-2
T = 5

n = 4
d = 1.7
Temp = 175

md = MD(1, 1, d*n, True)
md.add_molecules(generate_latice(n, d), Temp/119.7)
r, v = md.solve(dt, T, cache=False)

fN = 0
while os.path.exists(f"run/{fN}/"):
    fN += 1
os.mkdir(f"run/{fN}")
np.save(f"run/{fN}/r.npy", r)
np.save(f"run/{fN}/v.npy", v)
np.save(f"run/{fN}/ep.npy", md.ep)
np.save(f"run/{fN}/wallpass.npy", md.wallpass)
np.save(f"run/{fN}/check.npy", {"dt": dt, "T": T, "epsilon": md.epsilon, "sigma": md.sigma, "bboxlen": md.bboxlen, "p_bound": md.p_bound, "temp": md.temp})
np.save(f"run/{fN}/r_check.npy", md.r_init)

"""
md.velocity_autocorrelation()
msd = md.mean_squared_displacement(300)

D = msd[-1] / (6 * md.t[-1])

print(f"Diffusion coefficinet (msd): {D}")

plt.plot(md.t[:200], msd)
plt.title("Mean squared displacement")
plt.show()
"""