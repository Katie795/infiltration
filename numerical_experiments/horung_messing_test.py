"""Multple plots for mesh refinement for Horung-Messing problem."""
import numpy as np
import torch
from AI4Flood.richards_solver import *
from AI4Flood.generate_csv_files import *
from AI4Flood.richards_coefficients import u_exact_hm, kappa_hm, theta_hm
from AI4Flood.misc_functions import *
import time 

### Variables that can be changed ###
nx_array = torch.tensor([7, 15, 31, 63]) # number of elements in each edge of mesh

# THIS NOW NEEDS TO BE DIVISIBLE BY 4!!!
dt_array = torch.tensor([0.01, 0.005, 0.002, 0.001]) # timestep array
length = 1                           # domain is [0, 1] x [0, 1]
T = 0.01                             # total length of time
PC = True                            # predictor corrector method or L method
nlevels = [3, 4, 5, 6]

### Set up gpu specs ###
is_gpu = torch.cuda.is_available() # Check if GPU is available 
device = torch.device("cpu")
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)
print(f"Running on device {device}")

### iterates, lists and functions ###
L2_error = []
start = time.time()
with torch.no_grad():
    for dt in dt_array:
        print(f"dt is {dt}")
        L2_error_dx = []
        i = 0
        for nx in nx_array:
            print(f"nx is {nx}")
            t=0
            dx = length / nx
            x, y, xx, yy = generate_csv_files(nx, nx, length, length, name="horung_messsing")
            zz = torch.zeros_like(yy)

            u_init = u_exact_hm(x,y,0) # initial condition 
            uold = u_init.clone()

            if PC == True:
                model = Richards_PC(zz, dx, dx, theta_hm, kappa_hm, xx, yy)
            else:
                model = Richards_L(zz,  dx, dx, theta_hm, kappa_hm, xx, yy)

            while t < T:
                if PC == True:
                    u_new = model(uold, t, dt, ninner=50, boundary_type="Dirichlet", L=2, g=u_exact_hm, iteration=10, nlevel=3)
                else:
                    u_new = model(uold, t, dt, L=2, ninner=200, nouter=1000, boundary_type="Dirichlet", g=u_exact_hm)
                uold = u_new.clone()
                t += dt


                if t >= T:
                    print("t has reached the final time!")
                    break
                


            error = u_new - u_exact_hm(x, y, t) # spatial error

            plot_tensor(xx.cpu(), yy.cpu(), u_exact_hm(xx,yy, 0)[0,0,:,:].cpu().numpy(), "init_exact_solution")
            #plot_tensor(x.cpu(), y.cpu(), u_new[0,0,:,:].cpu().numpy(), "model_solution")
            plot_tensor(x.cpu(), y.cpu(), torch.abs(error)[0,0,:,:].cpu().numpy(), "model_exact_difference")

            L2_error_dx.append(torch.sqrt(torch.mean((u_new - u_exact_hm(x, y, t))**2)).cpu().numpy())
            i += 1

        L2_error.append(L2_error_dx)

#plot_tensor(xx.cpu(), yy.cpu(), u_exact_hm(xx, yy, T)[0,0,:,:].cpu().numpy() - u_exact_hm(xx, yy, t)[0,0,:,:].cpu().numpy(), "Exact_solution_error, T - t")
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

#for j in range(len(L2_error[0])):
#    ax2.loglog(dt_array.cpu().numpy(), L2_error[:,j], linestyle="dashed", marker=marker[j], label=rf"$\Delta x = {dx[j]:.6f}$")
#ax2.plot(dt, 0.001 * dt, color="black", linestyle="dotted", label=r"Slope $= -1$" )
#ax2.set_xlabel(r"$\tau$", fontsize=12)
#ax2.set_ylabel(r"$L^2$ error", fontsize=12)
#ax2.legend(fontsize=12)
plt.tight_layout()
if PC == True:
    plt.savefig(f"../plots/horung_messing_PC_T={T}.png")
else:
    plt.savefig(f"../plots/horung_messing_L_T={T}.png")
