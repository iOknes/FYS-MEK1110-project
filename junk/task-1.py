import numpy as np
import matplotlib.pyplot as plt
from ljp import LJP

r = np.linspace(0.9, 3, 1e4 + 1)
ljp = LJP(1, 1)

plt.plot(r, ljp(r), "g", label="Lennard-Jones potential for σ=1, ε=1")
plt.axhline(c="r", ls="--")
plt.axvline(0.95, c="b", ls="--")
plt.axvline(1.5, c="b", ls="--")
plt.grid()
plt.legend()
plt.show()
