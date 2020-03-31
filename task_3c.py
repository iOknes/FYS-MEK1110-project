import numpy as np
import matplotlib.pyplot as plt
from md import MD
from latice import generate_latice

if __name__ == "__main__":        
    atoms = generate_latice(3, 20/3)
    md = MD(1,1)
    md.track(atoms)
