"""Multple plots for mesh refinement for Horung-Messing problem."""
import numpy as np
import torch
from AI4Flood.richards_solver import *
from AI4Flood.generate_csv_files import *
from AI4Flood.richards_coefficients import u_exact_hm, kappa_hm, theta_hm
from AI4Flood.misc_functions import *

### Variables that can be changed ###
T = 0.001                                      # total length of time
nx_array = torch.tensor([10, 20, 40, 80, 160])# numer of elements in each edge of mesh
#dt_array = torch.tensor([0.001, 0.0005, 0.0002, 0.0001])            # timestep array
dt_array = torch.tensor([T, 0.5 * T, 0.2 * T, 0.1 * T])            # timestep array
#T_array = torch.tensor([0.001, 0.002, 0.004, 0.008])
length = 10                                    # domain is [0, 10] x [0, 10]
PC = False                                      # whether to use predictor corrector method

### Set up gpu specs ###
is_gpu = torch.cuda.is_available() # Check if GPU is available 
device = torch.device("cuda:0" if is_gpu else "cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)
print(f"Running on device {device}")


### iterates, lists and functions ###
niterations = []

L2_error = []

with torch.no_grad():
    for dt in dt_array:
        print(f"dt is {dt}")
        L2_error_dx = []
        for nx in nx_array:
            print(f"nx is {nx}")
            t=0
            dx = length / nx

            x, y, xx, yy = generate_csv_files(nx, nx, length, length, name="horung_messsing")

            zz = torch.zeros_like(yy)

            u_init = u_exact_hm(x,y,0).double() # initial condition
            uj   = u_init.clone()   # 
            uold = u_init.clone()

            if PC == True:
                model = Richards_PC(zz, dx, dx, theta_hm, kappa_hm)
            else:
                model = Richards_L(zz,  dx, dx, theta_hm, kappa_hm)

            while t < T:
                u_exact = u_exact_hm(xx,yy,t)
                u_new = model(uold, t, dt, ninner=10, boundary_type="Dirichlet", g=u_exact)
                t += dt
                if t >= T:
                    print("t has reached the final time!")
                    break
                uold = u_new.clone()

            error = u_new - u_exact_hm(x, y, t) # spatial error

            plot_tensor(xx.cpu(), yy.cpu(), u_exact_hm(xx,yy, t)[0,0,:,:].cpu().numpy(), "exact_solution")
            plot_tensor(x.cpu(), y.cpu(), u_new[0,0,:,:].cpu().numpy(), "model_solution")
            plot_tensor(x.cpu(), y.cpu(), torch.abs(error)[0,0,:,:].cpu().numpy(), "model_exact_difference")

            L2_error_dx.append(torch.sqrt(torch.mean((u_new - u_exact_hm(x, y, t))**2)).cpu().numpy())

        L2_error.append(L2_error_dx)

plot_tensor(xx.cpu(), yy.cpu(), u_exact_hm(xx, yy, T)[0,0,:,:].cpu().numpy() - u_exact_hm(xx, yy, 0)[0,0,:,:].cpu().numpy(), "Exact_solution_error, T - t")

dx = 1/(nx_array.cpu().numpy())
L2_error = np.array(L2_error)

### convergence plot ###
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
ax1.loglog(dx, L2_error[0], linestyle="dashed", marker="o", label=rf"$\Delta t = {dt_array[0]:.6f}$")
ax1.loglog(dx, L2_error[1], linestyle="dashed", marker="x", label=rf"$\Delta t = {dt_array[1]:.6f}$")
ax1.loglog(dx, L2_error[2], linestyle="dashed", marker="s", label=rf"$\Delta t = {dt_array[2]:.6f}$")
ax1.loglog(dx, L2_error[3], linestyle="dashed", marker="^", label=rf"$\Delta t = {dt_array[3]:.6f}$")
ax1.plot(dx, 0.1 * dx ** 2, color="black", linestyle="dotted", label=r"Slope $= -2$" )
ax1.set_xlabel(r"$\Delta x$", fontsize=12)
ax1.set_ylabel(r"$L^2$ error", fontsize=12)
ax1.legend(fontsize=12)

ax2.loglog(dt_array.cpu().numpy(), L2_error[:,0], linestyle="dashed", marker="o", label=rf"$\Delta x = {dx[0]}$")
ax2.loglog(dt_array.cpu().numpy(), L2_error[:,1], linestyle="dashed", marker="x", label=rf"$\Delta x = {dx[1]}$")
ax2.loglog(dt_array.cpu().numpy(), L2_error[:,2], linestyle="dashed", marker="s", label=rf"$\Delta x = {dx[2]}$")
ax2.loglog(dt_array.cpu().numpy(), L2_error[:,3], linestyle="dashed", marker=".", label=rf"$\Delta x = {dx[3]}$")
ax2.loglog(dt_array.cpu().numpy(), L2_error[:,4], linestyle="dashed", marker=">", label=rf"$\Delta x = {dx[4]}$")
#plt.plot(dt, 0.001 * dt, color="black", linestyle="dotted", label=r"Slope $= -1$" )
ax2.set_xlabel(r"$\tau$", fontsize=12)
ax2.set_ylabel(r"$L^2$ error", fontsize=12)
ax2.legend(fontsize=12)
plt.tight_layout()
if PC == True:
    plt.savefig(f"../plots/horung_messing_PC_T={T}.png")
else:
    plt.savefig(f"../plots/horung_messing_L_T={T}.png")
