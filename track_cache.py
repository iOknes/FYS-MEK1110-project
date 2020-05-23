import sys
import numpy as np
import matplotlib.pyplot as plt

fname = sys.argv[1]

r = np.load(f"cache/{fname}/r.npy")

trackfile = open("track.xyz", "w")

for i in r:
    trackfile.write(f"{len(i)}\ntype x y z\n")
    for j in i:
        trackfile.write(f"Ar {j[0]} {j[1]} {j[2]}\n")

trackfile.close()
