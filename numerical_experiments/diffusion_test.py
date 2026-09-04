"""Diffusion tests for richards solver"""

import numpy as np
import torch
from AI4Flood.richards_solver import *
from AI4Flood.generate_csv_files import *
from AI4Flood.misc_functions import *

torch.set_default_dtype(torch.float64)

### Variables that can be changed ###
nx_array = torch.tensor([7, 15, 31, 63]) # number of elements in each edge of mesh
## number of elements should be 2^n + 1 


# THIS NOW NEEDS TO BE DIVISIBLE BY 4!!!
dt_array = 0.01 * torch.tensor([0.1, 0.05, 0.02, 0.01]) # timestep array
length = 1                           # domain is [0, 1] x [0, 1]
T = 0.01 * 0.1                             # total length of time
KAPPA = False                        # whether to include nonlinear kappa
PC = True                            # predictor corrector method or L method
L_factor = 100

### Set up gpu specs ###
is_gpu = torch.cuda.is_available() # Check if GPU is available 
device = torch.device("cuda:0" if is_gpu else "cpu")
print(f"Running on device {device}")

### iterates, lists and functions ###
L2_error_dx = []
niterations = []
L2_error = []

def theta(s):
    return s

def Kappa1(s):
    return torch.ones_like(s)

def f1(x, y, t):
    return (2 * torch.pi ** 2 - 1 ) * torch.exp(-t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def Kappa2(s):
    return s**2 + 1

def f2(x, y, t):

    f = (2 * torch.pi ** 2 - 1 ) * torch.exp(-t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y) \
        + torch.pi ** 2 * torch.exp(-3 * t) * torch.sin(torch.pi * x) * ( torch.sin(torch.pi * x) ** 2 - 2 * torch.cos(torch.pi * x) ** 2  ) * torch.sin(torch.pi * y) ** 3  \
        + torch.pi ** 2 * torch.exp(-3 * t) * torch.sin(torch.pi * y) * ( torch.sin(torch.pi * y) ** 2 - 2 * torch.cos(torch.pi * y) ** 2  ) * torch.sin(torch.pi * x) ** 3 
    return f


def u_exact(x, y, t):
    u_exact = torch.exp(-torch.tensor(t)) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    return torch.unsqueeze(torch.unsqueeze(u_exact, 0), 0)

if KAPPA == True:
    kappa = Kappa2
    f = f2
else:
    kappa = Kappa1
    f = f1

with torch.no_grad():
    for dt in dt_array:
        L2_error_dx = []
        for nx in nx_array:
            t=0
            dx = length / nx

            x, y, xx, yy = generate_csv_files(nx, nx, length, length, name="testing")
            zz = torch.zeros_like(yy)

            u_init = u_exact(x,y,0) # initial condition
            uj   = u_init.clone()   # 
            uold = u_init.clone()

            if PC == True:
                model = Richards_PC2(zz, dx, dx, theta, kappa, xx, yy)
            else:
                model = Richards_L(zz, dx, dx, theta, kappa, xx, yy)

            while t < T:
                # TODO: how to we apply exact bcs to multigrids?
                
                if PC == True:
                    L = 5 #L_factor * nx * dt

                    u_new = model(uold, t, dt, ninner=10, boundary_type="Dirichlet", g=u_exact, f=f, L=L)
                else:
                    L = 5 #L_factor * nx * dt
   
                    u_new = model(uold, t, dt, L=10, ninner=100, nouter=200, boundary_type="Dirichlet", omega=0.5, g_old=g_old, g_new=g_new, f=f)
                t += dt
                if t >= T:
                    print("t has reached the final time!")
                    break
                uold = u_new.clone()

            error = u_new - u_exact(x, y, t) # spatial error
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
ax1.plot(dx, 0.1 * dx ** 2, color="black", linestyle="dotted", label=r"Slope $= -2$" )
ax1.set_xlabel(r"$\Delta x$", fontsize=12)
ax1.set_ylabel(r"$L^2$ error", fontsize=12)
ax1.legend(fontsize=12)
for j in range(len(L2_error[0])):
    ax2.loglog(dt_array.cpu().numpy(), L2_error[:,j], linestyle="dashed", marker=marker[j], label=rf"$\Delta x = {dx[j]:.6f}$")
ax2.plot(dt, 10 * dt, color="black", linestyle="dotted", label=r"Slope $= -1$" )
ax2.set_xlabel(r"$\tau$", fontsize=12)
ax2.set_ylabel(r"$L^2$ error", fontsize=12)
ax2.legend(fontsize=12)
plt.tight_layout()
if PC == True and KAPPA == True:
    plt.savefig(f"../plots/diffusion_PC_kappa.png")
elif PC == True and KAPPA == False:
    plt.savefig(f"../plots/diffusion_PC_linear.png")
elif PC == False and KAPPA == True:
    plt.savefig(f"../plots/diffusion_L_kappa.png")
else:
    plt.savefig(f"../plots/diffusion_L_linear.png")