"""Multple plots for mesh refinement for Horung-Messing problem."""
import numpy as np
import time 
from firedrake import *
from fem_solver import *

nx_array = np.array([7, 15, 31, 63]) # number of elements in each edge of mesh
nt_array = np.array([1, 2, 5, 10]) # timestep array
T = 0.01                             # total length of time

### iterates, lists and functions ###
L2_error = []
start = time.time()

#TODO: THIS TEST INVOLVES REGENERATING MESHES!!! CAN WE RUN IT WITHOUT REGENERATING THE MESHES?? 

for nt in nt_array:
    print(f"nt is {nt}")
    L2_error_dx = []
    for nx in nx_array:
        print(f"nx is {nx}")
        unew, error = Richards_FEM(nx, nt, T)
        L2_error_dx.append(error)
    L2_error.append(L2_error_dx)

end = time.time()
print(f"Time taken to run simulation: {end - start}")
dx = 1 / (nx_array.cpu().numpy())
L2_error = np.array(L2_error)
L2_error = np.array(L2_error)
print(f"L2 error for dt = {dt_array[0]} is:")
print(L2_error[0])
print(f"L2 error for dt = {dt_array[1]} is:")
print(L2_error[1])
print(f"L2 error for dt = {dt_array[2]} is:")
print(L2_error[2])
print(f"L2 error for dt = {dt_array[3]} is:")
print(L2_error[3])

### convergence plot ###
fig, ax1 = plt.subplots(1,1, figsize=(6,6))
marker = ["o", "x", "s", "^", ">"]
for i in range(len(L2_error)):
    ax1.loglog(dx, L2_error[i], linestyle="dashed", marker=marker[i], label=rf"$\Delta t = {dt_array[i]:.6f}$")
ax1.plot(dx, 0.005 * dx ** 2, color="black", linestyle="dotted", label=r"Slope $= -2$" )
ax1.set_xlabel(r"$\Delta x$", fontsize=12)
ax1.set_ylabel(r"$L^2$ error", fontsize=12)
ax1.legend(fontsize=12)
plt.tight_layout()
if PC == True:
    plt.savefig(f"../plots/horung_messing_FEM_T={T}.png")
else:
    plt.savefig(f"../plots/horung_messing_FEM_T={T}.png")
