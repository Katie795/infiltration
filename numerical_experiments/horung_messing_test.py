"""Multple plots for mesh refinement for Horung-Messing problem."""
import numpy as np
import torch
from AI4Flood.richards_solver import *
from AI4Flood.generate_csv_files import *
from AI4Flood.richards_coefficients import u_exact_hm, kappa_hm, theta_hm
from AI4Flood.misc_functions import *
from AI4Flood.boundary_conditions import BoundaryCondition
import time 
from matplotlib.ticker import ScalarFormatter

### spatial and temporal parameters ###
T = 0.1                                      # total length of time
nt_array = np.array([1, 2, 5, 10])           # number of timesteps
dt_array = np.array([0.1, 0.05, 0.02, 0.01]) # timestep array
nx_array = np.array([7, 15, 31, 63])         # number of elements in each edge of mesh
length = 10                                   # the length of the side of a square domain

PC = False               # predictor corrector method or L method
nlevels = [3, 4, 5, 6]  # levels of refinement


### Set up gpu specs ###
is_gpu = torch.cuda.is_available() # Check if GPU is available 
device = torch.device("cuda:0")
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)
print(f"Running on device {device}")

L2_error = []       # stores the error
start = time.time() # start timer

boundary = BoundaryCondition()

with torch.no_grad():
    for nx in nx_array:
        print(f"Evaluating on mesh {nx}")
        L2_error_nx = []
        dx = length / nx
        start_mesh = time.time() # start timer

        x, y, xx, yy = generate_csv_files(nx, nx, length, length, name="horung_messsing")
        end_mesh = time.time()

        zz = torch.zeros_like(yy)

        u_init = u_exact_hm(x,y,0) # initial condition 
        uold = u_init.clone()

        if PC == True:
            model = Richards_PC(xx, yy, dx, dx, theta_hm, kappa_hm, 3, drain_l=0, drain_r=0)
        else:
            model = Richards_L(xx, yy, dx, dx, theta_hm, kappa_hm, 0, drain_l=0, drain_r=0)

        i = 0
        for dt in dt_array:
            t = 0
            nt = nt_array[i]
            uold = u_init.clone()

            print(f"dt is {dt}")
            print(f"nt is {nt}")

            for n in range(nt):
                t = (n + 1) * dt
                g_old = u_exact_hm(xx,yy,t)
                g_new = u_exact_hm(xx,yy,t + dt)
                if PC == True:
                    u_new = model(uold, t, dt, boundary, boundary_type="Dirichlet", L=2, g=u_exact_hm, smoothing_iterations=1, MG_iterations=10, PC_iterations=100, nlevel=3)
                else:
                    u_new = model(uold, t, dt, L=2, ninner=100000, nouter=1000, boundary_type="Dirichlet", g_old=g_old, g_new=g_new)
                uold = u_new.clone()

            error = u_new - u_exact_hm(x, y, t) # spatial error

            L2_error_nx.append(torch.sqrt(torch.mean(error**2)).cpu().numpy())
            i += 1

        L2_error.append(L2_error_nx)

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
ax1.set_xticks([0.2, 0.5, 1.])
ax1.get_xaxis().set_major_formatter(ScalarFormatter())
ax1.set_xlabel(r"$\Delta x$", fontsize=12)
ax1.set_ylabel(r"$L^2$ error", fontsize=12)
ax1.legend(fontsize=12)
plt.tight_layout()
if PC == True:
    plt.savefig(f"../plots/hm_pc.png")
else:
    plt.savefig(f"../plots/hm_l.png")
