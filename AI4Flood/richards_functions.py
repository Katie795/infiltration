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

    x_coords = np.zeros((ny+1, nx+1))
    y_coords = np.zeros((ny+1, nx+1))
    for i in range(nx+1):
        for j in range(ny+1):
            x_coords[j][i] = dx * i
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

    return ans.unsqueeze(0).unsqueeze(0).to(torch.float64)


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

def apply_zero_bcs(u):
    u_new = u.clone()
    u_new[0,0,0,:] = torch.zeros_like(u_new[0][0][0][:])
    u_new[0, 0, -1, :] = torch.zeros_like(u_new[0][0][-1][:] )
    u_new[0,0, :, 0] = torch.zeros_like(u_new[0][0][:][0] )
    u_new[0, 0, :,-1] = torch.zeros_like(u_new[0][0][:][-1] )
    return u_new


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

    return ans.unsqueeze(0).unsqueeze(0).to(torch.float64)

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
    """u_exact in the Horung-Messing test.  Here
    u_exact(x,y,t) = -1/2 s                     if s < 0,
                   = - tan((e^s - 1)/(e^s + 1)) if s > 0.
    
    :args x: x coordinates
    :args y: y coordinates
    :args t: time
    """
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

    return uu.to(torch.float64)

def apply_bcs(u, g):
    """Applies a boundary condition g to a function u.
    
    :args u: original function over nx x ny grid to which the boundary condition is applied
    :args g: boundary function over nx+2 x ny+2 grid
    
    :returns uu: nx+2 x ny+2 function with u in interior cells and g on padded cells."""
    u_new = u.clone()

    u_new[0,0,0,:] = g[0,0,0,:] # insert u values in the interior 
    u_new[0,0,:,0] = g[0,0,:,0] # set Dirichlet boundary conditions 
    u_new[0,0,:,-1] = g[0,0,:,-1] # set Dirichlet boundary conditions
    u_new[0,0,-1,:] = g[0,0,-1,:]  # set Dirichlet boundary conditions 

    return u_new.to(torch.float64)

class AIRichards(nn.Module):
    def __init__(self, w1, wm, bias_initializer, x, y, theta, kappa, g, f=0, diag=1, ninner=10, nouter=10):
        super(AIRichards, self).__init__()
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

        self.g = g
        self.f = f

        self.diag = diag    

    def forward(self, uold, t, dt, ninner=10, nouter=10):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
        L = 2 # lipschitz constant for L method
        omega = 1 # damping parameter

        uj = uold.clone()

        bc_new = self.g(self.x, self.y, t + dt)
        uuold = dirichlet_boundary_condition(uold, bc_new)
        
        if self.f == 0:
            f_new = torch.tensor(0)
        else:
            f_new = self.f(self.x[1:-1, 1:-1], self.y[1:-1, 1:-1], t + dt)

        for j in range(nouter):

            uiplus = torch.zeros_like(uj, device=device) # set up jacobi iteration
            ui = uj.clone()

            uuj = dirichlet_boundary_condition(uj, bc_new)

            # Right hand side of Jacobi iteration - fixed
            b = L * uj + dt * f_new + self.cmm(self.theta(uuold) - self.theta(uuj))

            for i in range(ninner):
                uui = dirichlet_boundary_condition(ui, bc_new) # apply dirichlet boundary condition

                div_k_grad_ui = - 0.5 * (self.diff(self.kappa(uuj) * uui) + self.kappa(uj) * self.diff(uui) - ui * self.diff(self.kappa(uuj)))
                A = L * ui - dt * div_k_grad_ui
                D = self.diag * dt + L

                uiplus = (ui  + omega * (b - A) / D).clone() # Jacobi iteration
                uiplus = apply_bcs(uiplus, bc_new[:, :, 1:-1, 1:-1])
                
                assert torch.isinf(torch.sqrt(torch.mean((uiplus - ui)**2))) == False
                assert torch.isnan(torch.sqrt(torch.mean((uiplus - ui)**2))) == False

                if torch.sqrt(torch.mean((uiplus - ui)**2)) <= 1e-12:
                    print(f"Inner iterations converged after {i+1} iterations")
                    break

                if i == ninner - 1:
                    print(f"Inner iteration has not converged")
                    break

                ui = uiplus.clone()

            if torch.sqrt(torch.mean((uiplus - uj)**2)) <= 1e-12:
                print(f"Outer iterations converged after {i+1} iterations")
                break

            if j == nouter - 1:
                print(f"Outer iteration has not converged")
                break

            uj = uiplus.clone()        

        return uiplus    
    
class AIRichardsPC(nn.Module):
    def __init__(self, w1, wm, bias_initializer, x, y, theta, kappa, g, f=0, diag=1):
        super(AIRichardsPC, self).__init__()
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

        self.g = g
        self.f = f

        self.diag = diag    

    def forward(self, uold, t, dt, ninner=1, nouter=1):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
        L = 5 # lipschitz constant, should be 2.

        bc_new = self.g(self.x, self.y, t + dt)
        uuold = dirichlet_boundary_condition(uold, bc_new)

        if self.f == 0:
            f_new = torch.tensor(0)
        else:
            f_new = self.f(self.x[1:-1, 1:-1], self.y[1:-1, 1:-1], t + dt)

        ujplus = torch.zeros_like(uold, device=device)
        uj = uold.clone()

        for i in range(ninner):
            # predictor step
            uuj = dirichlet_boundary_condition(uj, u_exact(self.x, self.y, t + dt))
            # this is like a forward euler scheme
            div_k_grad_uj = - 0.5 * (self.diff(Kappa(uuj) * uuj) + Kappa(uj) * self.diff(uuj) - uj * self.diff(Kappa(uuj)))

            RHS = self.cmm(theta(uuold) - theta(uuj)) + dt * div_k_grad_uj + dt * f_new
            ujstar =  uj + 1 / L * RHS
            ujstar = apply_bcs(ujstar, bc_new[:, :, 1:-1, 1:-1])
            
            uujstar = dirichlet_boundary_condition(ujstar, u_exact(self.x, self.y,t+dt)) # apply boundary condition

            # corrector step
            ujpred = 0.5 * (uold + ujstar)
            uujpred = dirichlet_boundary_condition(ujpred, u_exact(self.x, self.y,t+dt))

            div_k_grad_uj_pred = - 0.5 * (self.diff(Kappa(uujpred) * uujpred) + Kappa(ujpred) * self.diff(uujpred) - ujpred * self.diff(Kappa(uujpred)))

            RHS_pred = self.cmm(theta(uuold) - theta(uujstar)) + dt * div_k_grad_uj_pred
            ujplus = (uj + 1 / L * RHS_pred).clone()
            ujplus = apply_bcs(ujplus, bc_new[:, :, 1:-1, 1:-1])
            
            if torch.sqrt(torch.mean((ujplus - uj)**2)) <= 1e-12:
                print(f"Ierations converged after {i+1} iterations")
                break

            uj = ujplus.clone()

        return ujplus


class AIRichardsNewton(nn.Module):
    def __init__(self, w1, wm, bias_initializer, x, y):
        super(AIRichardsNewton, self).__init__()
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

        self.diff.weight.data = w1
        self.cmm.weight.data = wm

        self.diff.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer

        self.loss = nn.MSELoss()
        self.x = x
        self.y = y

        

    def forward(self, uold, t, dt, imax=2):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""

        uj = uold.clone()
        bc_new = u_exact(self.x, self.y, t + dt)
        uuj  = dirichlet_boundary_condition(uj, bc_new)

        for i in range(imax):

            b = theta(uold) - theta(uj)

            uuj = dirichlet_boundary_condition(uj, bc_new)

            div_k_grad_uj = - 0.5 * (self.diff(Kappa(uuj) * uuj) + Kappa(uj) * self.diff(uuj) - uj * self.diff(Kappa(uuj))) 

            A =  dt * div_k_grad_uj 

            res_norm = torch.sqrt(torch.mean((b -A)**2)) 

            delta_u = torch.zeros_like(uj)


            if res_norm <= 1e-12:
                print(f"Newton converged after {i+1} iterations")
                break

            for _ in range(imax):

                #delta_bc = dirichlet_boundary_condition(delta_u, torch.zeros_like(uj))

                #K_u = Kappa(uj)  # frozen in Newton step

                div_delta = - 0.5 * (self.diff(Kappa(uuj) * uuj) + Kappa(uj) * self.diff(uuj) - uj * self.diff(Kappa(uuj))) 

                # Approx Jacobian action
                J_delta = Kappa_prime(uj) * delta_u - dt * div_delta

                res_lin = -(b - A) - J_delta

                delta_u = delta_u + res_lin

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