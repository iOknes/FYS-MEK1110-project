import os
import numpy as np

fN = 0
finished = False
while not finished:
    if os.path.exists(f"cache/{fN}"):
        print(f"{fN}:", np.load(f"cache/{fN}/check.npy", allow_pickle=True).item())
        fN += 1
    else:
        finished = True