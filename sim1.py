import numpy as np
import matplotlib.pyplot as plt
from world import World
from body import Body

#Define simulation parameters
dt = 1e-4
T = 5
N = int(T/dt)

#Define world object and bodies
w0 = World()
w0.addBody((0,0,0), (0,0,0), 0, 0)
w0.addBody((1.5,0,0), (0,0,0), 0, 0)

#Define and initialise tracker arrays
bodies = len(w0.bodies)
tracker_r = np.empty((N, bodies, w0.dim))
tracker_v = np.empty((N, bodies, w0.dim))

tracker_r[0] = np.array((w0.bodies[0].pos, w0.bodies[1].pos))
tracker_v[0] = np.array((w0.bodies[0].vel, w0.bodies[1].vel))

#Run simulation and add info to tracker arrays

for i in range(1, N):
    w0 *= dt
    for j in range(bodies):
        tracker_r[i,j] = w0.bodies[j].pos
        tracker_v[i,j] = w0.bodies[j].vel

#Output resulting velocity and position to files
outfile_r = open("ov_output_p.txt", "w")

for i in tracker_r:
    outfile_r.write(f"{len(i)}\n\n")
    for j in i:
        outfile_r.write("Ar")
        for k in j:
            outfile_r.write(f"\t{k}")
        outfile_r.write("\n")

outfile_r.close()

outfile_v = open("ov_output_v.txt", "w")

for i in tracker_v:
    outfile_v.write(f"{len(i)}\n\n")
    for j in i:
        outfile_v.write("Ar")
        for k in j:
            outfile_v.write(f"\t{k}")
        outfile_v.write("\n")

outfile_v.close()

rn = np.linalg.norm(tracker_r[:,0] - tracker_r[:,1], axis=1)
t = np.linspace(0, T, N)

plt.plot(t, rn)
plt.grid()
plt.show()
