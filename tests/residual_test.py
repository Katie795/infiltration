"""Multple plots for mesh refinement for Horung-Messing problem."""
import numpy as np
import torch
import torch.nn as nn
from AI4Flood.richards_solver import *

torch.set_default_dtype(torch.float64)

### Variables that can be changed ###
nx_array = torch.tensor([10, 20, 40, 80, 160])# numer of elements in each edge of mesh
length = 10                                    # domain is [0, 10] x [0, 10]
dt = 0.001


### Set up gpu specs ###
is_gpu = torch.cuda.is_available() # Check if GPU is available 
device = torch.device("cuda:0" if is_gpu else "cpu")
print(f"Running on device {device}")

### iterates, lists and functions ###
loss = nn.MSELoss()
bias_initializer = torch.tensor([0.0], device=device).double()
niterations = []

class CalculateResidual(nn.Module):
    def __init__(self, w1, wm, bias_initializer, x, y, theta, kappa, g, boundary_function, f=0, diag=1):
        super(CalculateResidual, self).__init__()
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.cmm.weight.data = wm

        self.diff.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer
        
        self.loss = nn.MSELoss()

        self.x = x
        self.y = y

        self.theta = theta
        self.kappa = kappa

        self.g = g # dirichlet boundary conditions
        self.boundary_function = boundary_function



    def forward(self, u, u_old, dt):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
        bc_new = self.g(self.x, self.y, t + dt)
        uu = self.boundary_function(u, bc_new)
        
        div_k_grad_ui = - 0.5 * (self.diff(self.kappa(uu) * uu) + self.kappa(u) * self.diff(uu) - u * self.diff(self.kappa(uu)))

        residual = div_k_grad_ui - (self.theta(u) - self.theta(u_old)) / dt

        return torch.max(residual) 

L2_error = []
with torch.no_grad():
    for nx in nx_array:
        t=0
        dx = length / nx

        generate_coords(nx, nx, name="horung_", length=length)
        generate_padded_coords(nx, nx, name="horung_padded_", length=length)
        x = torch.tensor(np.genfromtxt("../data/horung_x_coords.csv", delimiter=',')).double()
        y = torch.tensor(np.genfromtxt("../data/horung_y_coords.csv", delimiter=',')).double()
        x_padded = torch.tensor(np.genfromtxt("../data/horung_padded_x_coords.csv", delimiter=',')).double()
        y_padded = torch.tensor(np.genfromtxt("../data/horung_padded_y_coords.csv", delimiter=',')).double()

        u_old = u_exact(x,y,0).double() # initial condition
        u_ex = u_exact(x,y,dt).double() # initial condition

        w1 = torch.tensor([[[[-1/3/dx**2], [-1/3/dx**2] , [-1/3/dx**2]],
                    [[-1/3/dx**2], [8/3/dx**2], [-1/3/dx**2]],
                    [[-1/3/dx**2], [-1/3/dx**2] , [-1/3/dx**2]]]], device=device).double()
    
        
        wm = torch.tensor([[[[0.0], [0.0], [0.0]],
                        [[0.0], [1.0], [0.0]],
                        [[0.0], [0.0], [0.0]]]], device=device).double()

        w1 = torch.reshape(w1,(1,1,3,3)).double() # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
        wm = torch.reshape(wm,(1,1,3,3)).double()

        diag = 8/3/dx**2

        model = CalculateResidual(w1, wm, bias_initializer, x_padded, y_padded, theta, Kappa, g=u_exact, boundary_function=apply_bcs, diag=diag).to(device)
        residual = model(u_ex, u_old, dt)
        L2_error.append(residual)

dx = 1/(nx_array.cpu().numpy())
L2_error = np.array(L2_error)

print(L2_error)

### convergence plot ###
fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.loglog(dx, L2_error, linestyle="dashed", marker="o")
plt.plot(dx, 0.0001 * dx ** 2, color="black", linestyle="dotted", label=r"Slope $= -2$" )
ax.set_xlabel(r"$\Delta x$", fontsize=12)
ax.set_ylabel(r"$L^2$ error", fontsize=12)
plt.title(f"Residual test", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"../plots/residual_error.png")
