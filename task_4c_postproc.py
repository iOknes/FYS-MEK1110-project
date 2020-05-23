import numpy as np
import matplotlib.pyplot as plt
from md import MD
from latice import generate_latice
import os

r_len = 1000
msd_start = 150

msd = np.zeros(r_len - msd_start, dtype="float64")
D_vac = 0
D_msd = 0

fN = 0
while os.path.exists(f"run1/{fN}/"):
    ck = np.load(f"run1/{fN}/check.npy", allow_pickle=True).item()
    md = MD(ck["epsilon"], ck["sigma"], ck["bboxlen"], ck["p_bound"])
    md.r = np.load(f"run1/{fN}/r.npy")
    md.v = np.load(f"run1/{fN}/v.npy")
    md.ep = np.load(f"run1/{fN}/ep.npy")
    md.pN = len(md.r[0])
    md.wallpass = np.load(f"run1/{fN}/wallpass.npy")
    md.plot = False
    md.t = np.linspace(0, ck["T"], int(ck["T"]/ck["dt"]))

    D_vac += md.velocity_autocorrelation()
    msd_ = md.mean_squared_displacement(msd_start)
    msd += msd_
    D_msd += msd_[-60] / (6 * md.t[-60])

    fN += 1

N = fN + 1 

D_vac /= N
D_msd /= N
msd /= N

print(f"Diffusion coefficinet (vac): {D_vac}")
print(f"Diffusion coefficinet (msd): {D_msd}")

plt.plot(md.t[:r_len-msd_start], msd)
plt.title("Mean squared displacement")
plt.xlabel("Time")
plt.ylabel("Mean squared displacement")
plt.show()
