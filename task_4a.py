import numpy as np
import matplotlib.pyplot as plt
from md import MD
from latice import generate_latice

n = 3
d = 1.7
T = 174.4

md = MD(1, 1, d*n, True, rc=3)
md.add_molecules(generate_latice(n, d), T/119.7)
r, v = md.solve(1e-2, 2, plot=True)

#T = np.sum(md.v**2, axis=(1,2))/(3 * md.pN) * 119.7

T = np.zeros(md.v.shape[0], dtype="float64")

for i in range(len(md.v)):
    T[i] = 1 / (3 * md.pN) * np.sum(md.v[i]**2) * 119.7

print(f"T0: {T[0]}")
print(np.average(T[1:]))
print(np.average(np.linalg.norm(md.v, axis=(1,2))))

plt.plot(md.t, T, label="Temprature / K")
plt.xlabel("Time")
plt.ylabel("Temprature [K]")
plt.legend()
plt.show()
