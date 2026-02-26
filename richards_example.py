"""Getting a simple richards example working, comparing to Horung-Messing problem."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

nx = 100
ny = 100
dx = 0.1

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(f"Running on device {device}")

t = np.linspace(0, 2, 101)

def generate_coords(nx, ny, dx):
    """Generate csv files containing the x and y coordinate data for a uniform grid
    
    :args filename: csv file to generate coordintates from"""

    x_coords = np.zeros((ny, nx))
    y_coords = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            x_coords[j][i] = dx * i
            y_coords[j][i] = dx * ny - dx * (j + 1)

    np.savetxt("data/horung_x_coords.csv", x_coords, delimiter=",")
    np.savetxt("data/horung_y_coords.csv", y_coords, delimiter=",")
    return x_coords, y_coords


def theta(u):
    ans = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            if u[i][j] < 0:
                ans[i][j] = 0.5 * np.pi**2 - 2 * np.arctan(u[i][j])**2
            elif u[i][j] >= 0:
                ans[i][j] = 0.5 * np.pi**2
            else:
                print("Error in the theta calculation")
    return ans

def Kappa(u):
    ans = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            if u[i][j] < 0:
                ans[i][j] = 2 / (1 + u[i][j]**2)
            elif u[i][j] >= 0:
                ans[i][j] = 2
            else:
                print("Error in the theta calculation")
    return ans

x = np.genfromtxt("data/horung_x_coords.csv", delimiter=',')
y = np.genfromtxt("data/horung_y_coords.csv", delimiter=',')

def u_exact(x,y,t):
    ans = np.zeros((ny, nx))

    s = x - y - t
    for i in range(nx):
        for j in range(ny):
            if s[i][j] < 0:
                ans[i][j] = -0.5 * s[i][j]
            elif s[i][j] >= 0:
                ans[i][j] = - np.tan((np.exp(s[i][j])- 1)/(np.exp(s[i][j]) + 1))
            else:
                print("Error in the theta calculation")
    return torch.tensor(ans)

generate_coords(nx, ny, dx)



input_shape = (1, 1, ny, nx)
input_shape_pd = (1, 1, ny + 2, nx + 2)
u  = torch.zeros(input_shape, device=device)
uu = torch.zeros(input_shape_pd, device=device)

def dirichlet_boundary_condition(u, uu, g):
    ny = u.shape[2] 
    nx = u.shape[3]
    nny = uu.shape[2] # this is just ny + 2 (ny with padding)
    nnx = uu.shape[3] # this is just nx + 2 (with padding)
    ones = torch.ones(nny)

    uu[0,0,1:nny-1,1:nnx-1] = u[0,0,:,:] # insert u values in the interior
    uu[0,0,:,0]    = g * ones # set Dirichlet boundary conditions 
    uu[0,0,:,nx+1] = g * ones # set Dirichlet boundary conditions 
    uu[0,0,0,:]    = g * ones # set Dirichlet boundary conditions 
    uu[0,0,ny+1,:] = g * ones # set Dirichlet boundary conditions 
    return uu   

u_init = u_exact(x,y,0)


fig, ax = plt.subplots(1,1, figsize=(20,20))
height = ax.pcolormesh(x, y, u_init)
plt.savefig(f"plots/horung_test.png")

w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                    [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                    [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)
# Gradient in x 
w2 = torch.tensor([[[[1/(12*dx)], [0.0], [-1/(12*dx)]],
                    [[1/(3*dx)] , [0.0], [-1/(3*dx)]] ,
                    [[1/(12*dx)], [0.0], [-1/(12*dx)]]]], device=device)
# Gradient in y 
w3 = torch.tensor([[[[-1/(12*dx)], [-1/(3*dx)], [-1/(12*dx)]],
                    [[0.0]       , [0.0]      , [0.0]]       ,
                    [[1/(12*dx)] , [1/(3*dx)] , [1/(12*dx)]]]], device=device)
# Consistant mass matrix 
wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
                    [[0.11] ,  [0.44], [0.11]],
                    [[0.028], [0.11] , [0.028]]]], device=device)

class AIRichards(nn.Module):
    def __init__(self, w1, w2, w3, wm):
        super(AIRichards, self).__init__()
        self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.xadv.weight.data = w2
        self.yadv.weight.data = w3
        self.cmm.weight.data = wm

    def forward(self, u, uu, dt):
        # Taken from the paper!!! 
        L = 1

        uu = dirichlet_boundary_condition(u, uu)

        diff = 0.5 * self.diff(torch.mul(Kappa(u), u)) + Kappa(u) * self.diff(u) - u * self.diff(Kappa(u))

        u = u - dt * diff