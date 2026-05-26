import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

is_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if is_gpu else "cpu")

def plot_tensor(x, y, tensor, name):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    height = ax.pcolormesh(x, y, tensor)
    cb = plt.colorbar(height, label=name, ax=ax)
    plt.title(name)
    plt.savefig(f"../plots/{name}")

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

    np.savetxt(f"../data/{name}x_coords.csv", x_coords, delimiter=",")
    np.savetxt(f"../data/{name}y_coords.csv", y_coords, delimiter=",")

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

    np.savetxt(f"../data/{name}x_coords.csv", x_coords, delimiter=",")
    np.savetxt(f"../data/{name}y_coords.csv", y_coords, delimiter=",")

    return x_coords, y_coords

def theta(s):
    """theta(s) in the Horung-Messing test. Here
    theta(s) = pi**2/2 - 2 arctan(s)**2 if s < 0,
             = pi**2/2                  if s > 0.
    
    :args s: input where s = x - y - t
    """
    s = s[0, 0]  # still assumes (1,1,n,n)

    base = 0.5 * torch.pi**2
    atan_term = 2 * torch.atan(s)**2

    ans = torch.where(s < 0, base - atan_term, base)

    return ans.unsqueeze(0).unsqueeze(0).to(torch.float32)


def theta_prime(s):
    """theta'(s) in the Horung-Messing test. Needed for Newton's method. Here
    theta'(s) = -4 arctan(s) / (s**2 + 1) if s < 0,
              = 0                         if s > 0.
    
    :args s: input where s = x - y - t
    """
    ans = torch.zeros_like(s, device=device)
    news = s[0][0]

    for i in range(news[0]):
        for j in range(len(news[0])):
            if news[i][j] < 0:
                ans[i][j] =  - 4 * torch.atan(news[i][j])/ (news[i][j]**2 + 1)
            elif news[i][j] >= 0:
                ans[i][j] = 0
            else:
                print("Error in the theta prime calculation")
                break
    return torch.unsqueeze(torch.unsqueeze(ans, 0),0).to(torch.float32)



def Kappa(s):
    """Kappa(s) in the Horung-Messing test. Here
    Kappa(s) = 2 / (1 + s**2) if s < 0,
             = 2              if s > 0.
    
    :args s: input where s = x - y - t
    """
    s = s[0, 0]

    base = 2.0
    kappa_left = 2.0 / (1.0 + s**2)

    ans = torch.where(s < 0, kappa_left, base)

    return ans.unsqueeze(0).unsqueeze(0).to(torch.float32)

def Kappa_prime(s):
    """Kappa'(s) in the Horung-Messing test. Needed for Newton's method. Here
    Kappa'(s) = -4s / (1 + s**2)**2 if s < 0,
              = 0                   if s > 0.
    
    :args s: input where s = x - y - t
    """
    s = s[0, 0]

    base = 0
    kappa_prime_left = -4 * s / (1 + s**2)**2

    ans = torch.where(s < 0, kappa_prime_left, base)

    return ans.unsqueeze(0).unsqueeze(0).to(torch.float32)


def u_exact(x, y, t):
    s = x - y - t

    left = -0.5 * s
    right = -torch.tan(torch.tanh(s / 2))

    ans = torch.where(s < 0, left, right)

    return ans.unsqueeze(0).unsqueeze(0).to(torch.float32)

def dirichlet_boundary_condition(u, g):
    """Applies a boundary condition g to a function u.
    
    :args u: original function over nx x ny grid to which the boundary condition is applied
    :args g: boundary function over nx+2 x ny+2 grid
    
    :returns uu: nx+2 x ny+2 function with u in interior cells and g on padded cells."""

    ny = u.shape[2] 
    nx = u.shape[3]

    uu = torch.zeros_like(g)

    uu[0,0,1:nx+1,1:nx+1] = u[0,0,:,:] # insert u values in the interior 
    # left hand side 
    uu[0,0,:,0] = g[0,0,:,0] # set Dirichlet boundary conditions 
    # right hand side
    uu[0,0,:,nx+1] = g[0,0,:,nx+1] # set Dirichlet boundary conditions
    # top side
    uu[0,0,0,:] = g[0,0,0,:]  # set Dirichlet boundary conditions 
    # bottom side
    uu[0,0,ny+1,:] = g[0,0,ny+1,:]  # set Dirichlet boundary conditions 

    return uu   

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
        self.beta = beta
        self.diag = diag
        self.omega = omega
        self.loss = nn.MSELoss()

    def forward(self, x, y, uold, ujinput, t, dt, imax=100):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
        L = 2 # lipschitz constant 

        uuold   = dirichlet_boundary_condition(uold, u_exact(x,y,t))

        ujplus = torch.zeros_like(ujinput, device=device)
        uj = torch.zeros_like(ujinput, device=device)
        uj.data.copy_(ujinput.data)


        for i in range(imax):
            # Right hand side of Jacobi iteration

            uuj = dirichlet_boundary_condition(uj, u_exact(x,y,t+dt))
            div_k_grad_uj = - 0.5 * (self.diff(Kappa(uuj) * uuj) + Kappa(uj) * self.diff(uuj) - uj * self.diff(Kappa(uuj)))

            #A = dt * div_k_grad_uj
            #b = self.cmm(theta(uuold) - theta(uuj))

            #ujplus.data.copy_(uj.data  + self.omega * (b.data - A.data))

            # L METHOD!!!
            b = self.cmm(L * uuj + theta(uuold) - theta(uuj))
            #div_k_grad_uj = - 0.5 * (Kappa(uj) * self.diff(uuj)  + self.diff(Kappa(uuj) * uuj) - uj * self.diff(Kappa(uuj)) )  
            # Need to check this is the right iteration scheme
            A = self.cmm(L * uuj) + dt * div_k_grad_uj
            #D = self.diag
            ujplus.data.copy_(uj.data  + self.omega * (b.data - A.data))

            if self.loss(ujplus, uj) <= 1e-12:
                print(f"Jacobi iterations converged after {i+1} iterations")
                break

            print(f"L2 Difference between uj_plus and u_j is {self.loss(ujplus, uj)}")

            uj.data.copy_(ujplus.data)

        return ujplus, i 
    
class AIRichardsPC(nn.Module):
    def __init__(self, w1, wm, bias_initializer, beta, diag, length=1, nx=10, omega=1):
        super(AIRichardsPC, self).__init__()
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.cmm.weight.data = wm

        self.diff.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer
        
        #self.nx = nx
        #self.length = length
        #self.dx = length / nx
        #self.beta = beta
        #self.diag = diag
        #self.omega = omega
        self.loss = nn.MSELoss()

    def forward(self, x, y, uold, ujinput, t, dt, imax=2):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
        L = 5 # lipschitz constant, should be 2.

        uuold = dirichlet_boundary_condition(uold, u_exact(x,y,t))

        uj = torch.zeros_like(ujinput, device=device)
        uj.data.copy_(ujinput.data)


        for i in range(imax):
            # predictor step
            uuj = dirichlet_boundary_condition(uj, u_exact(x,y,t+dt))
            div_k_grad_uj = - 0.5 * (self.diff(Kappa(uuj) * uuj) + Kappa(uj) * self.diff(uuj) - uj * self.diff(Kappa(uuj)))

            ujstar = 1 / L * (self.cmm(theta(uuold) - theta(uuj) + L * uuj)) \
                     + 1 / L * (dt * div_k_grad_uj)
            
            uujstar = dirichlet_boundary_condition(ujstar, u_exact(x,y,t+dt)) # apply boundary condition

            # corrector step
            ujpred = 0.5 * (uold + ujstar)
            uujpred = 0.5 * (uuold + uujstar)
            div_k_grad_uj_pred = - 0.5 * (self.diff(Kappa(uujpred) * uujpred) + Kappa(ujpred) * self.diff(uujpred) - ujpred * self.diff(Kappa(uujpred)))

            ujplus = 1 / L * (self.cmm(theta(uuold) - theta(uujpred) + L * uujpred)) \
                     + 1 / L * (dt * div_k_grad_uj_pred)
            
            if self.loss(ujplus, uj) <= 1e-12:
                print(f"Jacobi iterations converged after {i+1} iterations")
                break

            print(f"L2 Difference between uj_plus and u_j is {self.loss(ujplus, uj)}")

            uj.data.copy_(ujplus.data)

        return ujplus, i 


class AIRichardsNewton(nn.Module):
    def __init__(self, w1, wm, bias_initializer, beta, diag, length=1, nx=10):
        super(AIRichardsNewton, self).__init__()
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.cmm.weight.data = wm

        self.diff.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer
        
        self.nx = nx
        self.length = length
        self.dx = length / nx
        self.diag = diag
        self.loss = nn.MSELoss()

    def forward(self, x, y, uold, ujinput, t, dt, imax=10, linmax=50, L=2):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""

        #ujplus = ujinput.clone()
        uj = ujinput.clone()
        uuj  = dirichlet_boundary_condition(uj, u_exact(x,y,t+dt))

        for i in range(imax):

            b = theta(uold) - theta(uj)
            kappa_k  = Kappa(uj)
            kappa_kk = Kappa(uuj)

            uuj = dirichlet_boundary_condition(uj, u_exact(x,y,t+dt))

            div_k_grad_uj = - 0.5 * (self.diff(kappa_kk * uuj) + kappa_k * self.diff(uuj) - uj * self.diff(kappa_kk)) 

            A =  dt * div_k_grad_uj 

            F = b - A
            res_norm = torch.norm(F)

            delta_u = torch.zeros_like(uj)


            if res_norm <= 1e-12:
                print(f"Newton converged after {i+1} iterations")
                break

            for _ in range(linmax):

                #delta_bc = dirichlet_boundary_condition(delta_u, torch.zeros_like(uj))

                #K_u = Kappa(uj)  # frozen in Newton step

                div_delta = - 0.5 * (self.diff(kappa_kk * uuj) + kappa_k * self.diff(uuj) - uj * self.diff(kappa_kk)) 

                # Approx Jacobian action
                J_delta = Kappa_prime(uj) * delta_u - dt * div_delta

                res_lin = -F - J_delta

                delta_u = delta_u + res_lin / (self.diag + 1e-12)

            uj = uj + delta_u

        return uj


if __name__ == "__main__":
    """Here we are plotting the output of these functions"""
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

    s = torch.unsqueeze(torch.unsqueeze(x - y - t, 0), 0) 



    plot_tensor(x, y, u_exact(x,y,t)[0,0,:,:].cpu(), "u_exact")
    plot_tensor(x, y, Kappa(s)[0,0,:,:].cpu(), "Kappa")
    plot_tensor(x, y, theta(s)[0,0,:,:].cpu(), "theta")

    uu = dirichlet_boundary_condition(u_exact(x,y,t), u_exact(x_padded,y_padded,t))
    plot_tensor(x_padded, y_padded, uu[0,0,:,:].cpu(), "dirichlet_bcs")
    plot_tensor(x, y, u_exact(x_padded,y_padded,t)[0,0,1:nx+2,1:nx+2].cpu() - uu[0,0,1:nx+2,1:nx+2].cpu(), "dirichlet_bcs_error")