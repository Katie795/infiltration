"""Multple plots for mesh refinement for Horung-Messing problem."""
import numpy as np
import time 
from firedrake import *
from fem_solver import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

### spatial and temporal parameters ###
T = 0.1                                      # total length of time
nt_array = np.array([1, 2, 5, 10])           # number of timesteps
dt_array = np.array([0.1, 0.05, 0.02, 0.01]) # timestep array
nx_array = np.array([7, 15, 31, 63])         # number of elements in each edge of mesh
#nx_array = np.array([63])

length = 10                                   # the length of the side of a square domain

L2_error = []       # stores the error
start = time.time() # start timer

for nx in nx_array:
    print(f"Evaluating on mesh {nx}")
    L2_error_nx = []
    mesh = SquareMesh(nx, nx, length, quadrilateral=True) # set up square mesh
    V = FunctionSpace(mesh, "CG", 1)                  # construct function space
    
    i = 0
    for dt in dt_array: # for all different time-steps
        nt = nt_array[i]
        print(f"dt is {dt}")
        print(f"nt is {nt}")

        unew, error = Richards_FEM(dt, nt, mesh, V, ninner=1000, nouter=1000) # solver
        L2_error_nx.append(error)
        i += 1 

    L2_error.append(L2_error_nx)
    print(f"L2 error for mesh {nx} is {L2_error}")


end = time.time() # end timer
print(f"Time taken to run simulation: {end - start}")

dx = length / nx_array
L2_error = np.array(L2_error)
print(f"L2_error matrix is {L2_error}")
for i in range(len(dt_array)):
    print(f"L2 error for dt = {dt_array[i]} is:")
    print(L2_error[:, i])

### convergence plot ###s
fig, ax1 = plt.subplots(1,1)
marker = ["o", "x", "s", "^", ">"]
for i in range(len(dt_array)):
    ax1.plot(dx, L2_error[:, i], linestyle="dashed", marker=marker[i], label=rf"$\Delta t = {dt_array[i]:.2f}$", linewidth=2)
ax1.plot(dx, 0.02 * dx ** 2, color="black", linestyle="dotted", label=r"$y= C(\Delta x)^2$")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.tick_params(labelsize="x-large")
ax1.set_xticks([0.2, 0.5, 1.])
ax1.get_xaxis().set_major_formatter(ScalarFormatter())
ax1.set_xlabel(r"$\Delta x$", fontsize=18)
ax1.set_ylabel(r"$L^2$ error", fontsize=18)
ax1.legend(fontsize=16)
plt.tight_layout()
plt.savefig(f"../plots/hm_fem.png")

