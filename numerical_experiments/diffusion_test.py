"""Diffusion tests for richards solver"""

import numpy as np
import torch
from AI4Flood.richards_solver import *
from AI4Flood.generate_csv_files import *
from AI4Flood.misc_functions import *

torch.set_default_dtype(torch.float64)

### Variables that can be changed ###
nx_array = torch.tensor([10, 20, 40])# numer of elements in each edge of mesh
dt_array = 0.1 * torch.tensor([0.01])            # timestep array
length = 1                                    # domain is [0, 10] x [0, 10]
T = 0.001                                      # total length of time
KAPPA = True                                  # whether to include nonlinear kappa
PC = False

### Set up gpu specs ###
is_gpu = torch.cuda.is_available() # Check if GPU is available 
device = torch.device("cuda:0" if is_gpu else "cpu")
print(f"Running on device {device}")

### iterates, lists and functions ###
bias_initializer = torch.tensor([0.0], device=device).double()
L2_error_dx = []
niterations = []

L2_error = []

#def theta(s):
#    return torch.ones_like(s)

def Kappa(s):
    return s**2 + 1

def Kappa1(s):
    return torch.ones_like(s)

def theta(s):
    return s

def u_exact(x, y, t):
    u_exact = torch.exp(-torch.tensor(t)) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    return torch.unsqueeze(torch.unsqueeze(u_exact, 0), 0)

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

            x, y, xx, yy = generate_csv_files(nx, nx, length, length, name="testing")

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
                model = Richards_L(w1, wm, bias_initializer, yy, dx, dx, theta, Kappa, diag).to(device)
            else:
                model = Richards_L(w1, wm, bias_initializer, yy,  dx, dx, theta, Kappa1, diag).to(device)

            g = torch.zeros((1, 1, nx+3, nx+3))

            while t < T:
                if KAPPA == True:
                    u_new = model(uold, t, dt, L=5, ninner=100, nouter=100, boundary_type="Dirichlet", omega=0.5, g=g, f=f, xx=xx, yy=yy)
                else:
                    u_new = model(uold, t, dt, L=5, ninner=100, nouter=100, boundary_type="Dirichlet", omega=0.5, g=g, f=f1, xx=xx, yy=yy)
                t += dt
                if t >= T:
                    print("t has reached the final time!")
                    break
                uold = u_new.clone()

            error = u_new - u_exact(x, y, t) # spatial error
            print("Model solution on boundary is:")
            print(u_new[0,0,0,:])
            print(u_new[0,0,-1,:])
            print(u_new[0,0,:,0])
            print(u_new[0,0,:,-1])

            print("Exact solution on boundary is:")
            print(u_exact(x, y, t)[0,0,0,:])
            print(u_exact(x, y, t)[0,0,-1,:])
            print(u_exact(x, y, t)[0,0,:,0])
            print(u_exact(x, y, t)[0,0,:,-1])


            plot_tensor(xx, yy, u_exact(xx,yy, t)[0,0,:,:].cpu().numpy(), "exact_solution")
            plot_tensor(x, y, u_new[0,0,:,:].cpu().numpy(), "model_solution")
            plot_tensor(x, y, torch.abs(error)[0,0,:,:].cpu().numpy(), "model_exact_difference")

            L2_error_dx.append(torch.sqrt(torch.mean((u_new - u_exact(x, y, t))**2)).cpu().numpy())

        L2_error.append(L2_error_dx)

dx = 1/(nx_array.cpu().numpy())
L2_error = np.array(L2_error)

### convergence plot ###
print(f"L2_error is {np.array(L2_error_dx)}")
marker = ["o", "x", "s", "^", ">"]

### convergence plot ###
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
for i in range(len(L2_error)):
    ax1.loglog(dx, L2_error[i], linestyle="dashed", marker=marker[i], label=rf"$\Delta t = {dt_array[i]:.6f}$")
ax1.plot(dx, 0.0001 * dx ** 2, color="black", linestyle="dotted", label=r"Slope $= -2$" )
ax1.set_xlabel(r"$\Delta x$", fontsize=12)
ax1.set_ylabel(r"$L^2$ error", fontsize=12)
ax1.legend(fontsize=12)
for j in range(len(L2_error[0])):
    ax2.loglog(dt_array.cpu().numpy(), L2_error[:,j], linestyle="dashed", marker=marker[j], label=rf"$\Delta x = {dx[j]}$")
ax2.plot(dt, 0.001 * dt, color="black", linestyle="dotted", label=r"Slope $= -1$" )
ax2.set_xlabel(r"$\tau$", fontsize=12)
ax2.set_ylabel(r"$L^2$ error", fontsize=12)
ax2.legend(fontsize=12)
plt.tight_layout()
if KAPPA == True:
    plt.savefig(f"../plots/errors_KAPPA.png")
else:
    plt.savefig(f"../plots/errors_NO_KAPPA.png")