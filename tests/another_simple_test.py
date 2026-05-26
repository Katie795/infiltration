"""Getting a simple richards example working, comparing to Horung-Messing problem."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from AI4Flood.richards_functions import dirichlet_boundary_condition, generate_coords, generate_padded_coords, plot_tensor

def u_exact(x, y, t):
    return x - y - t


nx_array = torch.tensor([20, 40, 80, 160, 320, 640])
nx = 10

length = 1


# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if is_gpu else "cpu")
print(f"Running on device {device}")

loss = nn.MSELoss()

bias_initializer = torch.tensor([0.0], device=device).to(torch.float32)

L2_error_dx = []

dx = length / nx

generate_coords(nx, nx, name="horung_", length=length)
generate_padded_coords(nx, nx, name="horung_padded_", length=length)
x = torch.tensor(np.genfromtxt("data/horung_x_coords.csv", delimiter=','))
y = torch.tensor(np.genfromtxt("data/horung_y_coords.csv", delimiter=','))
x_padded = torch.tensor(np.genfromtxt("data/horung_padded_x_coords.csv", delimiter=','))
y_padded = torch.tensor(np.genfromtxt("data/horung_padded_y_coords.csv", delimiter=','))

t = 0
u_old = u_exact(x, y, t)
plot_tensor(x_padded, y_padded, u_exact(x_padded,y_padded)[0,0,:,:].cpu().numpy(), "exact_solution")



class AITest(nn.Module):
    def __init__(self, w1, wm, bias_initializer, x, y):
        super(AITest, self).__init__()
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm.weight.data = wm
        self.cmm.bias.data = bias_initializer
        
        self.loss = nn.MSELoss()
        self.x = x
        self.y = y

    def forward(self, ujinput):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""

        ujplus = torch.zeros_like(ujinput, device=device)
        uj = torch.zeros_like(ujinput, device=device)
        uj.data.copy_(ujinput.data)

        # Right hand side of Jacobi iteration

        uuj = dirichlet_boundary_condition(uj, u_exact(self.x, self.y, t))

        ujplus.data.copy_(self.cpp(uuj.data)) 



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
        u_init_padded = u_exact(x_padded,y_padded).to(torch.float32)

        f_exact = f(x, y).to(torch.float32)
        f_exact_padded = f(x_padded, y_padded).to(torch.float32)
        
        input_shape = (1, 1, nx + 1, nx + 1)
        input_shape_pd = (1, 1, nx + 3, nx + 3)
        uj = torch.zeros(input_shape, device=device).to(torch.float32)
        g_exact = torch.zeros(input_shape_pd, device=device).to(torch.float32)
        uj.data.copy_(u_init.data)
        g_exact.data.copy_(u_init_padded.data)

        w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                    [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                    [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)
        
        w2 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                    [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                    [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)
        
        wm = torch.tensor([[[[0.0], [0.0], [0.0]],
                            [[0.0], [1.0], [0.0]],
                            [[0.0], [0.0], [0.0]]]], device=device)
        
        w1 = torch.reshape(w1,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
        w2 = torch.reshape(w2,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
        wm = torch.reshape(wm,(1,1,3,3))

        omega = 0.005 / nx**2

        model = AIRichards(w1, w2, wm, bias_initializer, length=length, nx=nx, omega=omega).to(device)
        u_new = model(x_padded, y_padded, uj, g_exact, f_exact, imax=100)

        # These should be closer together 
        error = u_new - u_exact(x, y)

        plot_tensor(x_padded, y_padded, u_exact(x_padded,y_padded)[0,0,:,:].cpu().numpy(), "exact_solution")
        plot_tensor(x, y, u_new[0,0,:,:].cpu().numpy(), "model_solution")
        plot_tensor(x, y, torch.abs(error)[0,0,:,:].cpu().numpy(), "model_exact_difference")

        L2_error_dx.append(loss(u_new, u_exact(x, y)).cpu().numpy())

### convergence plot ###
print(f"L2_error is {L2_error_dx}")
fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.loglog(1/(nx_array.cpu().numpy()), L2_error_dx, linestyle="dashed", marker="o")
plt.plot(1/(nx_array.cpu().numpy()), 0.0000000002 *(1/(nx_array.cpu().numpy()))**2, color="black", linestyle="dotted", label=r"Slope $= -2$" )
ax.set_xlabel(r"$\Delta x$", fontsize=18)
ax.set_ylabel(r"$L^2$ error", fontsize=18)
plt.title("Error against mesh-size", fontsize=22)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(f"plots/mesh_error_simple_test")

