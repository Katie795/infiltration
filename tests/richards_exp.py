"""Multple plots for mesh refinement for Horung-Messing problem."""
import numpy as np
import torch
import torch.nn as nn
from AI4Flood.richards_functions import *

torch.set_default_dtype(torch.float64)

### Variables that can be changed ###
nx_array = torch.tensor([10, 20, 40, 80, 160])# numer of elements in each edge of mesh
dt_array = torch.tensor([0.02, 0.01, 0.005, 0.0025, 0.00125])            # timestep array
length = 1                                    # domain is [0, 10] x [0, 10]
T = 0.02                                      # total length of time
KAPPA = True                                    # whether to include nonlinear kappa

### Set up gpu specs ###
is_gpu = torch.cuda.is_available() # Check if GPU is available 
device = torch.device("cuda:0" if is_gpu else "cpu")
print(f"Running on device {device}")

### iterates, lists and functions ###
loss = nn.MSELoss()
bias_initializer = torch.tensor([0.0], device=device).double()
L2_error_dx = []
niterations = []

L2_error = []

def theta(s):
    return torch.ones_like(s)

def Kappa(s):
    return s**2 + 1

def Kappa1(s):
    return torch.ones_like(s)

def theta(s):
    return s


def u_exact(x, y, t):
    u_exact = torch.exp(-torch.tensor(t)) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    return torch.unsqueeze(torch.unsqueeze(u_exact, 0), 0)

def g_zero(x, y, t):
    return torch.unsqueeze(torch.unsqueeze(torch.zeros_like(x), 0), 0)

def f(x, y, t):
    f = - torch.exp(-t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * (
        (1 - 2 * torch.pi ** 2) + torch.pi ** 2 * torch.exp(-2 * t ) * (
            torch.sin(torch.pi * y) ** 2 * (2 * torch.cos(torch.pi * x) ** 2 - torch.sin(torch.pi * y) ** 2)
            + torch.sin(torch.pi * x) ** 2 * (2 * torch.cos(torch.pi * y) ** 2 - torch.sin(torch.pi * x) ** 2)
        )
    )

    f2 = - 2 * torch.pi ** 2 * torch.exp(-t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * (
        torch.exp(-2*t) * (torch.sin(torch.pi * x )**2 + torch.sin(torch.pi * y )**2
                           - 3 * torch.sin(torch.pi * x )**2 * torch.sin(torch.pi * y )**2)
     - 1) - torch.exp(-t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    
    return f2

def f1(x, y, t):
    return (2 * torch.pi ** 2 - 1 ) * torch.exp(-t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

with torch.no_grad():
    for dt in dt_array:
        L2_error_dx = []
        for nx in nx_array:
            t=0
            dx = length / nx

            generate_coords(nx, nx, name="horung_", length=length)
            generate_padded_coords(nx, nx, name="horung_padded_", length=length)
            x = torch.tensor(np.genfromtxt("../data/horung_x_coords.csv", delimiter=',')).double()
            y = torch.tensor(np.genfromtxt("../data/horung_y_coords.csv", delimiter=',')).double()
            x_padded = torch.tensor(np.genfromtxt("../data/horung_padded_x_coords.csv", delimiter=',')).double()
            y_padded = torch.tensor(np.genfromtxt("../data/horung_padded_y_coords.csv", delimiter=',')).double()

            u_init = u_exact(x,y,0).double() # initial condition
            uj   = u_init.clone()   # 
            uold = u_init.clone()

            w1 = torch.tensor([[[[-1/3/dx**2], [-1/3/dx**2] , [-1/3/dx**2]],
                        [[-1/3/dx**2], [8/3/dx**2], [-1/3/dx**2]],
                        [[-1/3/dx**2], [-1/3/dx**2] , [-1/3/dx**2]]]], device=device).double()
            
            wm = torch.tensor([[[[0.0], [0.0], [0.0]],
                            [[0.0], [1.0], [0.0]],
                            [[0.0], [0.0], [0.0]]]], device=device).double()

            w1 = torch.reshape(w1,(1,1,3,3)).double() # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
            wm = torch.reshape(wm,(1,1,3,3)).double()

            diag = 8/3/dx**2

            if KAPPA == True:
                model = AIRichards(w1, wm, bias_initializer, x_padded, y_padded, theta, Kappa, g_zero, f, diag).to(device)
            else:
                model = AIRichards(w1, wm, bias_initializer, x_padded, y_padded, theta, Kappa1, g_zero, f1, diag).to(device).double()

            while t < T:
                u_new = model(uold, t, dt, imax=100)
                t += dt
                if t >= T:
                    print("t has reached the final time!")
                    break
                uold = u_new.clone()

            error = u_new - u_exact(x, y, t) # spatial error

            #plot_tensor(x_padded, y_padded, u_exact(x_padded,y_padded, t)[0,0,:,:].cpu().numpy(), "exact_solution")
            #plot_tensor(x, y, u_new[0,0,:,:].cpu().numpy(), "model_solution")
            #plot_tensor(x, y, torch.abs(error[0,0,:,:]).cpu().numpy(), "model_exact_difference_zerobc")

            L2_error_dx.append(loss(u_new, u_exact(x, y, t)).cpu().numpy())

        L2_error.append(L2_error_dx)

#plot_tensor(x_padded, y_padded, u_exact(x_padded,y_padded, T)[0,0,:,:].cpu().numpy() - u_exact(x_padded,y_padded, 0)[0,0,:,:].cpu().numpy(), "Exact_solution_error, T - t")

dx = 1/(nx_array.cpu().numpy())
L2_error = np.array(L2_error)

### convergence plot ###
print(f"L2_error is {np.array(L2_error_dx)}")


fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.loglog(dx, L2_error[0], linestyle="dashed", marker="o", label=rf"$\Delta t = {dt_array[0]:.4f}$")
plt.loglog(dx, L2_error[1], linestyle="dashed", marker="x", label=rf"$\Delta t = {dt_array[1]:.4f}$")
plt.loglog(dx, L2_error[2], linestyle="dashed", marker="s", label=rf"$\Delta t = {dt_array[2]:.4f}$")
plt.loglog(dx, L2_error[3], linestyle="dashed", marker="^", label=rf"$\Delta t = {dt_array[3]:.4f}$")
plt.plot(dx, 0.001 * dx ** 3, color="black", linestyle="dotted", label=r"Slope $= -3$" )
ax.set_xlabel(r"$\Delta x$", fontsize=12)
ax.set_ylabel(r"$L^2$ error", fontsize=12)
plt.title(f"Error against mesh-size, total time T = {T}", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
if KAPPA == True:
    plt.savefig(f"../plots/mesh_error_kappa")
else:
    plt.savefig(f"../plots/mesh_error_nokappa")

fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.loglog(dt_array.cpu().numpy(), L2_error[:, 0], linestyle="dashed", marker="o", label=rf"$\Delta x = {dx[0]}$")
plt.loglog(dt_array.cpu().numpy(), L2_error[:, 1], linestyle="dashed", marker="x", label=rf"$\Delta x = {dx[1]}$")
plt.loglog(dt_array.cpu().numpy(), L2_error[:, 2], linestyle="dashed", marker="s", label=rf"$\Delta x = {dx[2]}$")
plt.loglog(dt_array.cpu().numpy(), L2_error[:, 3], linestyle="dashed", marker=".", label=rf"$\Delta x = {dx[3]}$")
plt.loglog(dt_array.cpu().numpy(), L2_error[:, 4], linestyle="dashed", marker=">", label=rf"$\Delta x = {dx[4]}$")
plt.plot(dt, 0.001 * dt, color="black", linestyle="dotted", label=r"Slope $= -1$" )
ax.set_xlabel(r"$\Delta t$", fontsize=12)
ax.set_ylabel(r"$L^2$ error", fontsize=12)
plt.title(f"Error against time-step, total time T = {T}", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
if KAPPA == True:
    plt.savefig(f"../plots/time_error_kappa")
else:
    plt.savefig(f"../plots/time_error_nokappa")

#data = {"nx": nx_array.cpu().numpy(), "L2 error": np.array(L2_error_dx), "N iterations": np.array(niterations)}
#df = pd.DataFrame(data=data)

#print(df.to_latex(index=False))