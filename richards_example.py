"""Getting a simple richards example working, comparing to Horung-Messing problem."""
import numpy as np
import torch
import torch.nn as nn
from richards_functions import *


# h = 0.1 should give an L2 error of 0.01
nx_array = torch.tensor([10, 20, 40, 80])
dt = 0.01

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(f"Running on device {device}")

loss = nn.MSELoss()

bias_initializer = torch.tensor([0.0], device=device)
beta = 4                    # diagonal factor in mass term

class AIRichards(nn.Module):
    def __init__(self, w1, wm, bias_initializer, beta, diag, length=1, omega=1):
        super(AIRichards, self).__init__()
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.cmm.weight.data = wm

        self.diff.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer
        
        self.dx = length / nx
        self.beta = beta
        self.diag = diag
        self.omega = omega

    def forward(self, uold, uj, t, dt, imax=100):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
        # Taken from the paper!!! 
        # What values for L and D do we take?
        # L is the lipschitz constant!
        L = 2

        uuold   = dirichlet_boundary_condition(uold, u_exact(x_padded,y_padded,t+dt))
        uuj     = dirichlet_boundary_condition(uj, u_exact(x_padded,y_padded,t+dt))

        for i in range(imax):
            # Right hand side of Jacobi iteration

            uuj = dirichlet_boundary_condition(uj, u_exact(x_padded,y_padded,t+dt))

            div_k_grad_uj = - 0.5 * (Kappa(uuj, nx , nx) * self.diff(uuj)  + self.diff(Kappa(uuj, nx + 2, nx + 2 ) * uuj) - uj * self.diff(Kappa(uuj, nx + 2, nx + 2)))

            A = dt * div_k_grad_uj
            b = self.cmm(theta(uuold, nx+2, nx+2) - theta(uuj, nx+2, nx+2))

            ujplus = uj  + self.omega * (b - A)

            # L METHOD!!!
            #b = self.cmm(L * uuj + theta(uuold, nx+2, nx+2) - theta(uuj, nx+2, nx+2))
            #div_k_grad_uj = - 0.5 * (Kappa(uuj, nx , nx) * self.diff(uujplus)  + self.diff(Kappa(uuj, nx + 2, nx + 2 ) * uujplus) - ujplus * self.diff(Kappa(uuj, nx + 2, nx + 2)))   
            ## Need to check this is the right iteration scheme
            #A = self.cmm(L * uujplus) + dt * div_k_grad_uj
            #D = self.diag
            #ujplus = uj  + self.omega * (b - A) # residual loop. Is scale of D right? is A right? Is the balance of uuj and uujplus right?
            
            if loss(ujplus, uj) <= 1e-9:
                print(f"Jacobi iterations converged after {i+1} iterations")
                break

            print(f"L2 Difference between uj_plus and u_j is {loss(ujplus, uj)}")

            uj[:]= ujplus[:]

            print(f"L2 Difference between uj_plus and u_j is {loss(ujplus, uj)}")

        return ujplus  
itime = 0

L2_error = []




with torch.no_grad():
    for j in range(len(nx_array)):
        length = 10
        nx = nx_array[j]
        print(f"nx is {nx}")
        dx = length / nx
        print(f"dx is {dx}")
        generate_coords(nx, nx, name="horung_", length=length)
        generate_padded_coords(nx, nx, name="horung_padded_", length=length)
        x = torch.tensor(np.genfromtxt("data/horung_x_coords.csv", delimiter=','))
        y = torch.tensor(np.genfromtxt("data/horung_y_coords.csv", delimiter=','))
        x_padded = torch.tensor(np.genfromtxt("data/horung_padded_x_coords.csv", delimiter=','))
        y_padded = torch.tensor(np.genfromtxt("data/horung_padded_y_coords.csv", delimiter=','))

        u_init = u_exact(x,y,0)
        uold = u_init

        input_shape = (1, 1, nx + 1, nx + 1)
        input_shape_pd = (1, 1, nx + 3, nx + 3)
        uj = torch.zeros(input_shape, device=device)
        uj[0][0] = u_init


        w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                    [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                    [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)

        # Consistant mass matrix 
        wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
                            [[0.11] ,  [0.44], [0.11]],
                            [[0.028], [0.11] , [0.028]]]], device=device)

        w1 = torch.reshape(w1,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
        wm = torch.reshape(wm,(1,1,3,3))
        diag = -np.array(w1.cpu())[0,0,1,1]

        omega = 0.5 / nx

        model = AIRichards(w1, wm, bias_initializer, beta, diag, length=length, omega=omega).to(device)
        t=0

        u_new = model(uold, uj, t, dt, imax=500)
        error1 = u_new - u_exact(x, y, t+dt)

        plot_tensor(x_padded, y_padded, u_exact(x_padded,y_padded, t+dt)[0,0,:,:].cpu().numpy(), "exact_solution")
        plot_tensor(x, y, u_new[0,0,:,:].cpu().numpy(), "model_solution")
        plot_tensor(x, y, torch.abs(error1)[0,0,:,:].cpu().numpy(), "model_exact_difference")

        L2_error.append(loss(u_new, u_exact(x, y, t+dt)))

fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.loglog(1/nx_array.cpu().numpy(), L2_error, linestyle="dashed", marker="o", label=r"$\Delta t = 0.1$")
plt.plot(1/nx_array.cpu().numpy(), 0.2 *(1/nx_array.cpu().numpy())**2, color="black", linestyle="dotted", label=r"Slope $= -2$" )
ax.set_xlabel(r"$\Delta x$", fontsize=18)
ax.set_ylabel(r"$L^2$ error", fontsize=18)
plt.title("Error against mesh-size", fontsize=22)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(f"plots/mesh_error")