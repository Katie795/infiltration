"""Getting a simple richards example working, comparing to Horung-Messing problem."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

nx = 100
ny = 100
dx = 0.1
dt = 0.1
t_array = torch.arange(start=0, end=2, step=dt)

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(f"Running on device {device}")

def generate_coords(nx, ny, dx, shiftx=0, shifty=0, name=""):
    """Generate csv files containing the x and y coordinate data for a uniform grid
    
    :args filename: csv file to generate coordintates from"""

    x_coords = np.zeros((ny, nx))
    y_coords = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            x_coords[j][i] = dx * (i - shiftx)
            y_coords[j][i] = dx * ny - dx * ((j - shifty) + 1)

    np.savetxt(f"data/{name}x_coords.csv", x_coords, delimiter=",")
    np.savetxt(f"data/{name}y_coords.csv", y_coords, delimiter=",")
    return x_coords, y_coords

generate_coords(nx, ny, dx, name="horung_")
generate_coords(nx + 2, ny + 2, dx, shiftx=1, shifty=-1, name="horung_padded_")

def theta(u, nx, ny):
    ans = torch.zeros((ny, nx), device=device)
    print(f"u.shape is {u.shape}")
    newu = u[0][0]
    print(f"newu.shape is {newu.shape}")
    for i in range(nx):
        for j in range(ny):
            if newu[i][j] < 0:
                ans[i][j] = 0.5 * torch.pi**2 - 2 * torch.atan(newu[i][j])**2
            elif newu[i][j] >= 0:
                ans[i][j] = 0.5 * torch.pi**2
            else:
                print("Error in the theta calculation")
    print(f"Theta on device {ans.device}")
    return ans

def Kappa(u, nx, ny):
    ans = torch.zeros((ny, nx), device=device)
    newu = u[0][0]
    for i in range(nx):
        for j in range(ny):
            if newu[i][j] < 0:
                ans[i][j] = 2 / (1 + newu[i][j]**2)
            elif newu[i][j] >= 0:
                ans[i][j] = 2
            else:
                print("Error in the theta calculation")
    return ans

x = torch.tensor(np.genfromtxt("data/horung_x_coords.csv", delimiter=','))
y = torch.tensor(np.genfromtxt("data/horung_y_coords.csv", delimiter=','))
x_padded = torch.tensor(np.genfromtxt("data/horung_padded_x_coords.csv", delimiter=','))
y_padded = torch.tensor(np.genfromtxt("data/horung_padded_y_coords.csv", delimiter=','))

def u_exact(x,y,t):
    
    ans = torch.zeros((len(y), len(x)), device=device)

    s = x - y - t
    for i in range(nx):
        for j in range(ny):
            if s[i][j] < 0:
                ans[i][j] = -0.5 * s[i][j]
            elif s[i][j] >= 0:
                ans[i][j] = - torch.tan((torch.exp(s[i][j])- 1)/(torch.exp(s[i][j]) + 1))
            else:
                print("Error in the theta calculation")
    return ans



input_shape = (1, 1, ny, nx)
input_shape_pd = (1, 1, ny + 2, nx + 2)
uold   = torch.zeros(input_shape, device=device)
uuold  = torch.zeros(input_shape_pd, device=device)
uj     = torch.zeros(input_shape, device=device)
ujplus = torch.zeros(input_shape, device=device)
uuj    = torch.zeros(input_shape_pd, device=device)

print(f"u_old shape is {uold.shape}")

def dirichlet_boundary_condition(u, uu, g):
    print(f"Shape of u is {u.shape}")
    ny = u.shape[2] 
    nx = u.shape[3]
    nny = uu.shape[2] # this is just ny + 2 (ny with padding)
    nnx = uu.shape[3] # this is just nx + 2 (with padding)
    ones = torch.ones(nny, device=device)
    print(uu[0,0,:,0].shape)
    print(g.shape)
    print(ones.shape)



    uu[0,0,1:nny-1,1:nnx-1] = u[0,0,:,:] # insert u values in the interior
    uu[0,0,:,0]    = g[:,0] * ones # set Dirichlet boundary conditions 
    uu[0,0,:,nx+1] = g[:,nx+1] * ones # set Dirichlet boundary conditions 
    uu[0,0,0,:]    = g[0,:] * ones # set Dirichlet boundary conditions 
    uu[0,0,ny+1,:] = g[ny+1,:] * ones # set Dirichlet boundary conditions 
    return uu   

u_init = u_exact(x,y,0)


fig, ax = plt.subplots(1,1, figsize=(20,20))
height = ax.pcolormesh(x, y, u_init.cpu().numpy())
plt.title("Analytical solution for horung test")
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

    def forward(self, uold, uuold, uuj, uj, t, dt, imax=10):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
        # Taken from the paper!!! 
        # Here we
        L = 1
        print(f"Length of x_padded is {len(x_padded)}")

        uuold     = dirichlet_boundary_condition(uold, uuold, u_exact(x_padded,y_padded,t))
        uuj       = dirichlet_boundary_condition(uj, uuj, u_exact(x_padded,y_padded,t))
        uujplus   = dirichlet_boundary_condition(uj, uuj, u_exact(x_padded,y_padded,t))

        print(f"uuold.device is {uuold.device}")
        print(f"uuj is of device {uujplus.device}")
        ujplus = uj

        

        for i in range(2):
            # Right hand side of Jacobi iteration
            b = L * uj + theta(uold, nx, ny) - theta(uj, nx, ny)

            print(self.xadv(uuj))



            ## Need to check this is the right iteration scheme
            A = L * ujplus + dt * 0.5 * self.diff(torch.mul(Kappa(uuj, nx + 2, ny + 2), uujplus)) + Kappa(uuj) * self.diff(uujplus) - uujplus * self.diff(Kappa(uuj))
            D = self.diag
            
            ujplus = uj  + 1 / D * (b - A) # residual loop. Is scale of D right? is A right? Is the balance of uuj and uujplus right?
            uj = ujplus

        return ujplus
    

model = AIRichards(w1, w2, w3, wm).to(device)
itime = 0

L2_error = []

with torch.no_grad():
    u_old = torch.unsqueeze(torch.unsqueeze(u_init, 0), 0)
    uj = torch.unsqueeze(torch.unsqueeze(u_init, 0), 0)
    for itime in range(1, len(t_array)+1):
        unew = model(u_old, uuold, uuj, uj, t_array[itime], dt)

        itime +=1

        L2_error.append(torch.linalg.matrix_norm(unew, u_exact(x,y,t_array[itime])))


print(L2_error)