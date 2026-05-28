"""Multple plots for mesh refinement for Horung-Messing problem."""
import numpy as np
import torch
import torch.nn as nn
from AI4Flood.richards_functions import *


### Variables that can be changed ###
nx_array = torch.tensor([10, 20, 40, 80, 160]) # numer of elements in each edge of mesh
dt_array = torch.tensor([0.002, 0.001, 0.0005, 0.00025])              # timestep array
length = 10                                    # domain is [0, 10] x [0, 10]
T = 0.002                                      # total length of time
PC = False                                      # whether to use predictor corrector method

### Set up gpu specs ###
is_gpu = torch.cuda.is_available() # Check if GPU is available 
device = torch.device("cuda:0" if is_gpu else "cpu")
print(f"Running on device {device}")

### iterates, lists and functions ###
loss = nn.MSELoss()
bias_initializer = torch.tensor([0.0], device=device)
L2_error_dx = []
niterations = []

L2_error = []

with torch.no_grad():
    for dt in dt_array:
        L2_error_dx = []
        for nx in nx_array:
            t=0
            dx = length / nx

            generate_coords(nx, nx, name="horung_", length=length)
            generate_padded_coords(nx, nx, name="horung_padded_", length=length)
            x = torch.tensor(np.genfromtxt("../data/horung_x_coords.csv", delimiter=','))
            y = torch.tensor(np.genfromtxt("../data/horung_y_coords.csv", delimiter=','))
            x_padded = torch.tensor(np.genfromtxt("../data/horung_padded_x_coords.csv", delimiter=','))
            y_padded = torch.tensor(np.genfromtxt("../data/horung_padded_y_coords.csv", delimiter=','))

            u_init = u_exact(x,y,0) # initial condition
            uj   = u_init.clone()   # 
            uold = u_init.clone()

            w1 = torch.tensor([[[[-1/3/dx**2], [-1/3/dx**2] , [-1/3/dx**2]],
                        [[-1/3/dx**2], [8/3/dx**2], [-1/3/dx**2]],
                        [[-1/3/dx**2], [-1/3/dx**2] , [-1/3/dx**2]]]], device=device)
            
            # Consistant mass matrix 
            #wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
            #                    [[0.11] ,  [0.44], [0.11]],
            #                    [[0.028], [0.11] , [0.028]]]], device=device)
            
            wm = torch.tensor([[[[0.0], [0.0], [0.0]],
                            [[0.0], [1.0], [0.0]],
                            [[0.0], [0.0], [0.0]]]], device=device)

            w1 = torch.reshape(w1,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
            wm = torch.reshape(wm,(1,1,3,3))

            diag = -8/3/dx**2
            if PC == True:
                model = AIRichardsPC(w1, wm, bias_initializer, x_padded, y_padded).to(device)
            else:
                model = AIRichards(w1, wm, bias_initializer, x_padded, y_padded).to(device)

            while t < T:
                u_new, nit = model(uold, t, dt, imax=100)
                t += dt
                if t >= T:
                    print("t has reached the final time!")
                    break
                uold = u_new.clone()

            error = u_new - u_exact(x, y, t) # spatial error

            #plot_tensor(x_padded, y_padded, u_exact(x_padded,y_padded, t)[0,0,:,:].cpu().numpy(), "exact_solution")
            #plot_tensor(x, y, u_new[0,0,:,:].cpu().numpy(), "model_solution")
            #plot_tensor(x, y, torch.abs(error)[0,0,:,:].cpu().numpy(), "model_exact_difference")

            L2_error_dx.append(loss(u_new, u_exact(x, y, t)).cpu().numpy())
            niterations.append(nit)

        L2_error.append(L2_error_dx)

plot_tensor(x_padded, y_padded, u_exact(x_padded,y_padded, T)[0,0,:,:].cpu().numpy() - u_exact(x_padded,y_padded, 0)[0,0,:,:].cpu().numpy(), "Exact_solution_error, T - t")

dx = 1/(nx_array.cpu().numpy())

print(L2_error)

### convergence plot ###
print(f"L2_error is {np.array(L2_error_dx)}")
fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.loglog(dx, L2_error[0], linestyle="dashed", marker="o", label=rf"$\Delta t = {dt_array[0]:.4f}$")
plt.loglog(dx, L2_error[1], linestyle="dashed", marker="x", label=rf"$\Delta t = {dt_array[1]:.4f}$")
plt.loglog(dx, L2_error[2], linestyle="dashed", marker="s", label=rf"$\Delta t = {dt_array[2]:.4f}$")
plt.loglog(dx, L2_error[3], linestyle="dashed", marker="^", label=rf"$\Delta t = {dt_array[3]:.4f}$")
plt.plot(dx, 0.001 * dx ** 2, color="black", linestyle="dotted", label=r"Slope $= -2$" )
ax.set_xlabel(r"$\Delta x$", fontsize=12)
ax.set_ylabel(r"$L^2$ error", fontsize=12)
plt.title(f"Error against mesh-size, total time T = {T}", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
if PC == True:
    plt.savefig(f"../plots/mesh_error_PC")
else:
    plt.savefig(f"../plots/mesh_error")

'''
fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.loglog(dt_array.cpu().numpy(), L2_error[:][0], linestyle="dashed", marker="o", label=rf"$\Delta x = {dx[0]}$")
plt.loglog(dt_array.cpu().numpy(), L2_error[:][1], linestyle="dashed", marker="x", label=rf"$\Delta x = {dx[1]}$")
plt.loglog(dt_array.cpu().numpy(), L2_error[:][2], linestyle="dashed", marker="s", label=rf"$\Delta x = {dx[2]}$")
plt.loglog(dt_array.cpu().numpy(), L2_error[:][3], linestyle="dashed", marker=".", label=rf"$\Delta x = {dx[3]}$")
plt.loglog(dt_array.cpu().numpy(), L2_error[:][4], linestyle="dashed", marker=">", label=rf"$\Delta x = {dx[4]}$")
plt.plot(dx, 0.001 * dt, color="black", linestyle="dotted", label=r"Slope $= -1$" )
ax.set_xlabel(r"$\Delta t$", fontsize=12)
ax.set_ylabel(r"$L^2$ error", fontsize=12)
plt.title(f"Error against time-step, total time T = {T}", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"../plots/time_error")
'''
#data = {"nx": nx_array.cpu().numpy(), "L2 error": np.array(L2_error_dx), "N iterations": np.array(niterations)}
#df = pd.DataFrame(data=data)

#print(df.to_latex(index=False))