import numpy as np
import matplotlib.pyplot as plt
from potential import LJP

u = LJP(1,1)

x = np.linspace(0.9,3.1,1001)
y = u(x)

plt.plot(x, y, label="LJP: epsilon = 1, sigma = 1")
plt.title("Lennard-Jones potential for Sigma = 1, Epsilon = 1")
plt.xlabel("Distance")
plt.ylabel("Potential")
plt.legend()
plt.grid()
plt.show()
