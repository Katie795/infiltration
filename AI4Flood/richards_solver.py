import torch
import torch.nn as nn

is_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if is_gpu else "cpu")


class Richards(nn.Module):
    def __init__(self, w1, wm, bias_initializer, zz, dx, dy, theta, kappa, diag, f, drain_l, drain_r):
        super().__init__()

        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff.weight.data = w1
        self.cmm.weight.data = wm
        self.diff.bias.data = bias_initializer
        self.cmm.bias.data = bias_initializer

        self.zz = zz
        self.z = zz[1:-1, 1:-1]
        self.dx = dx
        self.dy = dy
        self.drain_l = drain_l
        self.drain_r = drain_r

        self.theta = theta
        self.kappa = kappa

        self.f = f
        self.diag = diag

    def create_padding(self, u):
        """Creates a zero tensor with padding"""
        ny = u.shape[2]
        nx = u.shape[3]
        input_shape = (1, 1, ny + 2, nx + 2) 
        u_new = torch.zeros(input_shape)
        return u_new

    def find_closest_index(self, value, dx):
        return torch.round(value / dx).int()


    def apply_dirichlet_bcs(self, u, g=1):
        """Applies a boundary condition g to a function u.
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function with u in interior cells and g on padded cells."""

        # creating tensor
        u_new = self.create_padding(u)

        u_new[0,0,1:-1, 1:-1] = u # fill interior

        u_new[0,0,0,:] = g[0,0,0,:]    # top dirichlet
        u_new[0,0,:,0] = g[0,0,:,0]    # left dirichlet
        u_new[0,0,:,-1] = g[0,0,:,-1]  # right dirichlet
        u_new[0,0,-1,:] = g[0,0,-1,:]  # bottom dirichlet

        return u_new
        
    def apply_mixed_bcs(self, u, g=1, kappa=1, q=2):
        """Applies a boundary condition g to a function u.
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function"""

        # creating tensor
        u_new = self.create_padding(u)

        u_new[0,0,1:-1, 1:-1] = u # fill interior

        cut1 = self.find_closest_index(self.drain_l, self.dx)
        cut2 = self.find_closest_index(self.drain_r, self.dx)

        u_new[0,0,0,:] = g[0,0,0,:]                # top dirichlet
        u_new[0,0,:,0]       = u_new[0,0,:,1]      # left homogeneous neumann  
        u_new[0,0,:,-1]      = u_new[0,0,:,-2]     # right homogeneous neumann
        u_new[0,0,-1, :cut1] = u_new[0,0,-2,:cut1] # bottom left homogeneous neumann
        u_new[0,0,-1, cut2:] = u_new[0,0,-2,cut2:] # bottom right homogeneous neumann
        u_new[0,0,-1, cut1:cut1] = u_new[0, 0, -2, cut1:cut1] - self.dy * (q / kappa(u_new[:, :, -2, cut1:cut1]) + 1)# bottom inhomogeneous neumann

        return u_new

    def apply_neumann_bcs(self, u, g=1, kappa=1, q=2):
        """Applies a boundary condition g to a function u. Used when the soil is not saturated
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function"""

        # creating tensor
        u_new = self.create_padding(u)

        u_new[0,0,1:-1, 1:-1] = u # fill interior

        cut1 = self.find_closest_index(self.drain_l, self.dx)
        cut2 = self.find_closest_index(self.drain_r, self.dx)

        u_new[0,0,0,:]   = u_new[0,0,0,:] + self.dy * (q / kappa(u_new[:,:,0,:] ) + 1) # inhomogeneous neumann in m 
        u_new[0,0,:,0]   = u_new[0,0,:,1]          # homogeneous neumann  
        u_new[0,0,:,-1]  = u_new[0,0,:,-2]         # homogeneous neumann
        u_new[0,0,-1, :cut1] = u_new[0,0,-2,:cut1] # homogeneous neumann
        u_new[0,0,-1, cut2:] = u_new[0,0,-2,cut2:] # homogeneous neumann

        id_active = u_new[0,0,-1, cut1:cut2] > 0 
        id_inactive = u_new[0,0,-1, cut1:cut2] <= 0

        u_new[0,0,-1, cut1:cut2][id_active] = 0 * u_new[0,0,-1, cut1:cut2][id_active] # set to zero!
        u_new[0,0,-1, cut1:cut2][id_inactive] = u_new[0,0,-2, cut1:cut2][id_inactive] # neumann bcs

        u_new[0,0,-1, cut1:cut1] = u_new[0, 0, -2, cut1:cut1] - self.dy * (q / kappa(u_new[:, :, -2, cut1:cut1]) + 1)# inhomogeneous neumann

        return u_new


class Richards_L(Richards):
    def __init__(self, w1, wm, bias_initializer, zz, dx, dy,  theta, kappa, f=0, diag=1):
         super().__init__(w1, wm, bias_initializer, zz, dx, dy, theta, kappa, f=0, diag=1)

    def forward(self, uold, t, dt, boundary_type, ninner=10, nouter=10, L=2, omega=1, q=1, g=1):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""

        # Choose which boundary function to use
        if boundary_type == "Dirichlet":
            boundary_function = self.apply_dirichlet_bcs
        elif boundary_type == "Mixed":
            boundary_function = self.apply_mixed_bcs
        elif boundary_type == "Neumann":
            boundary_function = self.apply_neumann_bcs
        else:
            raise Exception("Need to specify Dirichlet, Mixed or Neumann boundary conditions")

        uj = uold.clone()

        uuold = boundary_function(uold, g=g, kappa=self.kappa, q=q)
        
        if self.f == 0:
            f_new = torch.tensor(0)
        else:
            f_new = self.f(self.x[1:-1, 1:-1], self.y[1:-1, 1:-1], t + dt)

        for j in range(nouter):

            uiplus = torch.zeros_like(uj, device=device) # set up jacobi iteration
            ui = uj.clone()

            uuj = boundary_function(uj, g=g, kappa=self.kappa, q=q)

            # Right hand side of Jacobi iteration - fixed
            b = L * uj + dt * f_new + self.cmm(self.theta(uuold) - self.theta(uuj))

            for i in range(ninner):
                uui = boundary_function(ui, g=g, kappa=self.kappa, q=q) # apply dirichlet boundary condition
    
                div_k_grad_ui = - 0.5 * (self.diff(self.kappa(uuj + self.z) * uui)  \
                                         + self.kappa(uj + self.z) * self.diff(uui) \
                                            - ui * self.diff(self.kappa(uuj + self.z)))

                A = L * ui - dt * div_k_grad_ui
                D = self.diag * dt + L

                uiplus = (ui  + omega * (b - A) / D).clone() # Jacobi iteration
                
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
    
class Richards_PC(Richards):
    def __init__(self, w1, wm, bias_initializer, zz, dx, dy, theta, kappa, diag, f, drain_l, drain_r):
         super().__init__(w1, wm, bias_initializer, zz, dx, dy, theta, kappa, diag, f, drain_l, drain_r)

    def forward(self, uold, t, dt, boundary_type, ninner=1, q=1, g=1):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
        L = 5 # lipschitz constant, should be 2.
 
        if boundary_type == "Dirichlet":
            boundary_function = self.apply_dirichlet_bcs
        elif boundary_type == "Mixed":
            boundary_function = self.apply_mixed_bcs
        elif boundary_type == "Neumann":
            boundary_function = self.apply_neumann_bcs
        else:
            raise Exception("Need to specify Dirichlet, Mixed or Neumann boundary conditions")
            
        uuold = boundary_function(uold, g=g, kappa=self.kappa, q=q)

        if self.f == 0:
            f_new = torch.tensor(0)
        else:
            f_new = self.f(self.x[1:-1, 1:-1], self.y[1:-1, 1:-1], t + dt)

        ujplus = torch.zeros_like(uold, device=device)
        uj = uold.clone()



        for i in range(ninner):
            # predictor step
            uuj = boundary_function(uj, g=g, kappa=self.kappa, q=q)
            # this is like a forward euler scheme
            div_k_grad_uj = - 0.5 * (self.diff(self.kappa(uuj) * (uuj + self.zz)) \
                                     + self.kappa(uj) * self.diff(uuj + self.zz) \
                                          - (uj + self.z)* self.diff(self.kappa(uuj)))

            RHS = self.cmm(self.theta(uuold) - self.theta(uuj)) + dt * div_k_grad_uj + dt * f_new
            ujstar =  uj + 1 / L * RHS
            
            uujstar = boundary_function(ujstar, g=g, kappa=self.kappa, q=q)

            # corrector step
            ujpred = 0.5 * (uold + ujstar)
            uujpred = boundary_function(ujpred, g=g, kappa=self.kappa, q=q)

            div_k_grad_uj_pred = - 0.5 * (self.diff(self.kappa(uujpred) * (uujpred + self.zz)) \
                                          + self.kappa(ujpred) * self.diff(uujpred + self.zz) \
                                            - (ujpred + self.z)* self.diff(self.kappa(uujpred)))

            RHS_pred = self.cmm(self.theta(uuold) - self.theta(uujstar)) + dt * div_k_grad_uj_pred
            ujplus = (uj + 1 / L * RHS_pred).clone()
            #ujplus = self.apply_malago_infiltration_bcs(ujplus, q)

            assert torch.isinf(torch.sqrt(torch.mean((ujplus - uj)**2))) == False
            assert torch.isnan(torch.sqrt(torch.mean((ujplus - uj)**2))) == False
            
            if torch.sqrt(torch.mean((ujplus - uj)**2)) <= 1e-12:
                print(f"Iterations converged after {i+1} iterations")
                break

            uj = ujplus.clone()

        return ujplus
