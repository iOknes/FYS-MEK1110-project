import numpy as np
import matplotlib.pyplot as plt
from md import MD
from task_3c import generate_latice

md = MD(1,1)
md.add_molecules(generate_latice(4,1.7), np.zeros(256))
r, v = md.solve(1e-2, 2, 1)
