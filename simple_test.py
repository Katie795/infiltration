"""Getting a simple richards example working, comparing to Horung-Messing problem."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from richards_functions import dirichlet_boundary_condition, generate_coords, generate_padded_coords, plot_tensor

# h = 0.1 should give an L2 error of 0.01
nx_array = torch.tensor([20, 40, 80, 160])

length = 1 # domain is [0, 10] x [0, 10]

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if is_gpu else "cpu")
print(f"Running on device {device}")

loss = nn.MSELoss()

bias_initializer = torch.tensor([0.0], device=device).to(torch.float32)
beta = 4                    # diagonal factor in mass term

L2_error_dx = []

def u_exact(x,y):

    ans = torch.sin(torch.pi * x ) * torch.sin(torch.pi * y)
    return torch.unsqueeze(torch.unsqueeze(ans, 0), 0).to(device)

def f(x,y):

    ans = 2 * torch.pi **2 *torch.sin(torch.pi * x ) * torch.sin(torch.pi * y)
    return torch.unsqueeze(torch.unsqueeze(ans, 0), 0).to(torch.float32).to(device)

def kappa(u):
    return torch.ones_like(u).to(torch.float32)

class AIRichards(nn.Module):
    def __init__(self, w1, wm, bias_initializer, beta, diag, length=1, nx=10, omega=1):
        super(AIRichards, self).__init__()
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.cmm.weight.data = wm

        self.diff.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer
        
        self.nx = nx
        self.length = length
        self.dx = length / nx
        self.omega = omega
        self.loss = nn.MSELoss()

    def forward(self, ujinput, g, f, imax=100):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""

        ujplus = torch.zeros_like(ujinput, device=device)
        uj = torch.zeros_like(ujinput, device=device)
        uj.data.copy_(ujinput.data)
        b = self.cmm(f)

        for i in range(imax):
            # Right hand side of Jacobi iteration

            uuj = dirichlet_boundary_condition(uj, g)
            A = - self.diff(uuj)  

            ujplus.data.copy_(uj.data  + self.omega * (b.data - A.data))

            if self.loss(ujplus, uj) <= 1e-9:
                print(f"Jacobi iterations converged after {i+1} iterations")
                break

            if torch.isinf(self.loss(ujplus, uj)):
                print(f"Jacobi iterations not converges and produced an inf value")
                break

            print(f"L2 Difference between uj_plus and u_j is {self.loss(ujplus, uj)}")

            uj.data.copy_(ujplus.data)

        return ujplus  

with torch.no_grad():
    for nx in nx_array:
        dx = length / nx

        generate_coords(nx, nx, name="horung_", length=length)
        generate_padded_coords(nx, nx, name="horung_padded_", length=length)
        x = torch.tensor(np.genfromtxt("data/horung_x_coords.csv", delimiter=','))
        y = torch.tensor(np.genfromtxt("data/horung_y_coords.csv", delimiter=','))
        x_padded = torch.tensor(np.genfromtxt("data/horung_padded_x_coords.csv", delimiter=','))
        y_padded = torch.tensor(np.genfromtxt("data/horung_padded_y_coords.csv", delimiter=','))

        u_init = u_exact(x,y).to(torch.float32)
        f_exact = f(x_padded, y_padded).to(torch.float32)
        
        input_shape = (1, 1, nx + 1, nx + 1)
        input_shape_pd = (1, 1, nx + 3, nx + 3)
        uj = torch.zeros(input_shape, device=device).to(torch.float32)
        g_exact = torch.zeros(input_shape_pd, device=device).to(torch.float32)
        uj.data.copy_(u_init.data)


        w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                    [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                    [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)
        wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
                            [[0.11] ,  [0.44], [0.11]],
                            [[0.028], [0.11] , [0.028]]]], device=device)
        
        #wm = torch.tensor([[[[dx**2/36], [dx**2/9] , [dx**2/36]],
        #        [[dx**2/9] ,  [dx**2*4/9], [dx**2/9]],
        #        [[dx**2/36], [dx**2/9] , [dx**2/36]]]], device=device)


        w1 = torch.reshape(w1,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
        wm = torch.reshape(wm,(1,1,3,3))
        diag = -np.array(w1.cpu())[0,0,1,1]

        omega = 0.05 / nx

        model = AIRichards(w1, wm, bias_initializer, beta, diag, length=length, nx=nx, omega=omega).to(device)
        u_new = model(uj, g_exact, f_exact, imax=1000)

        # These shouls be closer together 
        error1 = u_new - u_exact(x, y)

        plot_tensor(x_padded, y_padded, u_exact(x_padded,y_padded)[0,0,:,:].cpu().numpy(), "exact_solution")
        plot_tensor(x, y, u_new[0,0,:,:].cpu().numpy(), "model_solution")
        plot_tensor(x, y, torch.abs(error1)[0,0,:,:].cpu().numpy(), "model_exact_difference")

        L2_error_dx.append(loss(u_new, u_exact(x, y)).cpu().numpy())

### convergence plot ###
print(f"L2_error is {L2_error_dx}")
fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.loglog(1/(nx_array.cpu().numpy()), L2_error_dx, linestyle="dashed", marker="o")
plt.plot(1/(nx_array.cpu().numpy()), 0.2 *(1/(nx_array.cpu().numpy()))**2, color="black", linestyle="dotted", label=r"Slope $= -2$" )
ax.set_xlabel(r"$\Delta x$", fontsize=18)
ax.set_ylabel(r"$L^2$ error", fontsize=18)
plt.title("Error against mesh-size", fontsize=22)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(f"plots/mesh_error_simple_test")