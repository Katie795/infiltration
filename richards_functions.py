import torch
import numpy as np
import matplotlib.pyplot as plt

is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")

def plot_tensor(x, y, tensor, name):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    height = ax.pcolormesh(x, y, tensor)
    cb = plt.colorbar(height, label=name, ax=ax)
    plt.title(name)
    plt.savefig(f"plots/{name}")

def generate_coords(nx, ny, name="", length=1):
    """Generate csv files containing the x and y coordinate data for a uniform grid
    
    :args filename: csv file to generate coordintates from"""

    dx = length / nx
    print(f"dx is {dx}")

    x_coords = np.zeros((ny+1, nx+1))
    y_coords = np.zeros((ny+1, nx+1))
    for i in range(nx+1):
        for j in range(ny+1):
            x_coords[j][i] = dx * (i)
            y_coords[j][i] = dx * ny - dx * j

    np.savetxt(f"data/{name}x_coords.csv", x_coords, delimiter=",")
    np.savetxt(f"data/{name}y_coords.csv", y_coords, delimiter=",")

    return x_coords, y_coords

def generate_padded_coords(nx, ny, name="", length=1):
    """Generate csv files containing the x and y coordinate data for a uniform grid
    
    :args filename: csv file to generate coordintates from"""

    dx = length / nx

    x_coords = np.zeros((ny+3, nx+3))
    y_coords = np.zeros((ny+3, nx+3))
    for i in range(nx+3):
        for j in range(ny+3):
            x_coords[j][i] = dx * (i - 1)
            y_coords[j][i] = dx * ny - dx * (j-1) 

    np.savetxt(f"data/{name}x_coords.csv", x_coords, delimiter=",")
    np.savetxt(f"data/{name}y_coords.csv", y_coords, delimiter=",")

    return x_coords, y_coords

def theta(s, nx, ny):
    ans = torch.zeros((ny+1, nx+1), device=device)
    news = s[0][0]

    for i in range(nx+1):
        for j in range(ny+1):
            if news[i][j] < 0:
                ans[i][j] = 0.5 * torch.pi**2 - 2 * torch.atan(news[i][j])**2
            elif news[i][j] >= 0:
                ans[i][j] = 0.5 * torch.pi**2
            else:
                print("Error in the theta calculation")
                break

    return torch.unsqueeze(torch.unsqueeze(ans, 0),0)

def theta_prime(s, nx, ny):
    ans = torch.zeros((ny+1, nx+1), device=device)
    news = s[0][0]

    for i in range(nx+1):
        for j in range(ny+1):
            if news[i][j] < 0:
                ans[i][j] =  - 4 * torch.atan(news[i][j])/ (news[i][j]**2 + 1)
            elif news[i][j] >= 0:
                ans[i][j] = 0
            else:
                print("Error in the theta calculation")
                break
    return torch.unsqueeze(torch.unsqueeze(ans, 0),0)


def Kappa(s, nx, ny):
    ans = torch.zeros((ny+1, nx+1), device=device)
    news = s[0][0]
    for i in range(nx+1):
        for j in range(ny+1):
            if news[i][j] < 0:
                ans[i][j] = 2 / (1 + news[i][j]**2)
            elif news[i][j] >= 0:
                ans[i][j] = 2
            else:
                print("Error in the theta calculation")
    return torch.unsqueeze(torch.unsqueeze(ans, 0),0)

def Kappa_prime(s, nx, ny):
    ans = torch.zeros((ny+1, nx+1), device=device)
    news = s[0][0]
    for i in range(nx+1):
        for j in range(ny+1):
            if news[i][j] < 0:
                ans[i][j] = -4*x / (1 + news[i][j]**2)**2
            elif news[i][j] >= 0:
                ans[i][j] = 0
            else:
                print("Error in the theta calculation")
    return torch.unsqueeze(torch.unsqueeze(ans, 0),0)

def u_exact(x,y,t):
    
    ans = torch.zeros((len(y), len(x)), device=device)

    s = x - y - t
    for i in range(len(x)):
        for j in range(len(y)):
            if s[j][i] < 0:
                ans[j][i] = -0.5 * s[j][i]
            elif s[j][i] >= 0:
                ans[j][i] = - torch.tan((torch.exp(s[j][i])- 1)/(torch.exp(s[j][i]) + 1))
            else:
                print("Error in the theta calculation")
    return torch.unsqueeze(torch.unsqueeze(ans, 0), 0)

def dirichlet_boundary_condition(u, g):

    ny = u.shape[2] 
    nx = u.shape[3]

    uu = torch.zeros_like(g)

    uu[0,0,1:nx+1,1:nx+1] = u[0,0,:,:] # insert u values in the interior 
    # left hand side 
    uu[0,0,:,0] = g[0,0,:,1] # set Dirichlet boundary conditions 
    # right hand side
    uu[0,0,:,nx+1] = g[0,0,:,nx] # set Dirichlet boundary conditions
    # top side
    uu[0,0,0,:] = g[0,0,1,:]  # set Dirichlet boundary conditions 
    # bottom side
    uu[0,0,ny+1,:] = g[0,0,ny,:]  # set Dirichlet boundary conditions 

    return uu   

if __name__ == "__main__":
    """Here we are plotting the output of these functions!
    DO A SIMPLE TEST TO SEE IF IT WORKS!"""
    length = 1
    nx = 10
    t = 0

    input_shape_pd = (1, 1, nx + 3, nx + 3)
    uu = torch.zeros(input_shape_pd)

    # Checking that the coordinates are correct - do we really need an extra 1?
    generate_coords(nx, nx, name="horung_", length=length)
    generate_padded_coords(nx, nx, name="horung_padded_", length=length)
    x = torch.tensor(np.genfromtxt("data/horung_x_coords.csv", delimiter=','))
    y = torch.tensor(np.genfromtxt("data/horung_y_coords.csv", delimiter=','))
    x_padded = torch.tensor(np.genfromtxt("data/horung_padded_x_coords.csv", delimiter=','))
    y_padded = torch.tensor(np.genfromtxt("data/horung_padded_y_coords.csv", delimiter=','))



    plot_tensor(x, y, u_exact(x,y,t)[0,0,:,:], "u_exact")
    plot_tensor(x, y, Kappa(x-y-t, nx, nx)[0,0,:,:], "Kappa")
    plot_tensor(x, y, theta(x-y-t, nx, nx)[0,0,:,:], "theta")

    uu = dirichlet_boundary_condition(u_exact(x,y,t), uu, u_exact(x_padded,y_padded,t))
    plot_tensor(x_padded, y_padded, uu[0,0,:,:], "dirichlet_bcs")





