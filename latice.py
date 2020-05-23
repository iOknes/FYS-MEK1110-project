import numpy as np

def generate_cell(i, j, k, d):
    return d * np.array([
        [i, j, k],
        [i, 0.5+j, 0.5+k],
        [0.5+i, j, 0.5+k],
        [0.5+i, 0.5+j, k]
    ])

def generate_latice(n, d):
    atoms = np.empty((4*n**3, 3), dtype="float64")
    index = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                atoms[index:index + 4] = generate_cell(i,j,k,d)
                index += 4
    return atoms     