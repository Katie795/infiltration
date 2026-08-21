import torch
import torch.nn as nn
from AI4Flood.operators import get_weights_linear_2D
from AI4Flood.misc_functions import jacobi, jacobi2

class Richards(nn.Module):
    def __init__(self, zz, dx, dy, theta, kappa, drain_l=0, drain_r=0):
        super().__init__()

        self.zz = zz
        self.z = zz[1:-1, 1:-1]
        self.dx = dx
        self.dy = dy
        self.drain_l = drain_l
        self.drain_r = drain_r

        self.theta = theta
        self.kappa = kappa

        [w_diff, wm1, diag, bias] = get_weights_linear_2D(self.dx)
        
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff.weight.data = w_diff
        self.cmm.weight.data = wm1
        self.diff.bias.data = bias
        self.cmm.bias.data = bias
        self.diag = diag

        self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0) # restriction
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest")) # prolongation



    def apply_dirichlet_bcs(self, u, g=1, kappa=1, q=0):
        """Applies a boundary condition g to a function u.
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function with u in interior cells and g on padded cells."""

        # adding padding to tensor
        u_new = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)

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

        # adding padding to tensor
        u_new = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)

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

        # adding padding to tensor
        u_new = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)

        cut1 = self.find_closest_index(self.drain_l, self.dx)
        cut2 = self.find_closest_index(self.drain_r, self.dx)

        u_new[0,0,0,:]   = u_new[0,0,0,:] + self.dy * (q / kappa(u_new[:,:,0,:] ) + 1) # inhomogeneous neumann in m 
        u_new[0,0,:,0]   = u_new[0,0,:,1]          # homogeneous neumann  
        u_new[0,0,:,-1]  = u_new[0,0,:,-2]         # homogeneous neumann
        u_new[0,0,-1, :cut1] = u_new[0,0,-2,:cut1] # homogeneous neumann
        u_new[0,0,-1, cut2:] = u_new[0,0,-2,cut2:] # homogeneous neumann

        id_active = u_new[0,0,-2, cut1:cut2] >= 0 
        id_inactive = u_new[0,0,-2, cut1:cut2] < 0

        u_new[0,0,-1, cut1:cut2][id_active] = 0 * u_new[0,0,-1, cut1:cut2][id_active] # set to zero!
        u_new[0,0,-1, cut1:cut2][id_inactive] = u_new[0,0,-2, cut1:cut2][id_inactive] # neumann bcs

        #u_new[0,0,-1, cut1:cut2] = u_new[0, 0, -2, cut1:cut2] - self.dy * (q / kappa(u_new[:, :, -2, cut1:cut2]) + 1)# inhomogeneous neumann

        return u_new

    def apply_conditional_bcs(self, u, g=1, kappa=1, q=2):
        """Applies conditional boundary conditions onto the soil 
                
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: here g is a nx + 2 tensor corresponding to the height of the water!!!
        
        :returns uu: nx+2 x ny+2 function"""


        # creating tensor
        # adding padding to tensor
        u_new = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)

        cut1 = self.find_closest_index(self.drain_l, self.dx)
        cut2 = self.find_closest_index(self.drain_r, self.dx)

        saturated   = u_new[0,0, 1, :] >= 0 
        unsaturated = u_new[0,0,1,:] < 0

        # TODO: does flux correspond to per unit space or per unit time?

        u_new[0,0,0,:][unsaturated] = u_new[0,0,0,:][unsaturated] + self.dy * (g[unsaturated] / kappa(u_new[:,:,0,:] )[0, 0, unsaturated] + 1) # inhomogeneous neumann in m 
        u_new[0,0,0,:][saturated] = g[saturated] 
        u_new[0,0,:,0]   = u_new[0,0,:,1]          # homogeneous neumann  
        u_new[0,0,:,-1]  = u_new[0,0,:,-2]         # homogeneous neumann
        u_new[0,0,-1, :cut1] = u_new[0,0,-2,:cut1] + self.dx # homogeneous neumann
        u_new[0,0,-1, cut2:] = u_new[0,0,-2,cut2:] + self.dx # homogeneous neumann

        id_active = u_new[0,0,-2, cut1:cut2] >= 0 # why are these 
        id_inactive = u_new[0,0,-2, cut1:cut2] < 0

        u_new[0,0,-1, cut1:cut2][id_active] = 0 * u_new[0,0,-1, cut1:cut2][id_active] # set to zero seepage occurs
        u_new[0,0,-1, cut1:cut2][id_inactive] = u_new[0,0,-2, cut1:cut2][id_inactive] + self.dx # neumann bcs - no seepage

        #u_new[0,0,-1, cut1:cut2][id_active] = u_new[0, 0, -2, cut1:cut2] - self.dy * (q / kappa(u_new[:, :, -2, cut1:cut2]) + 1)# inhomogeneous neumann
        # This is not how seeoage boundary conditions are implemented!!
        return u_new

class Richards_L(Richards):
    def __init__(self, zz, dx, dy,  theta, kappa, drain_l=0, drain_r=0):
         super().__init__(zz, dx, dy, theta, kappa, drain_l=0, drain_r=0)

    def forward(self, uold, t, dt, boundary_type, ninner=10, nouter=10, L=2, omega=1, q=1, g_old=1, g_new=1, f=0, xx=1, yy=1):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""

        # Choose which boundary function to use
        if boundary_type == "Dirichlet":
            boundary_function = self.apply_dirichlet_bcs
        elif boundary_type == "Mixed":
            boundary_function = self.apply_mixed_bcs
        elif boundary_type == "Neumann":
            boundary_function = self.apply_neumann_bcs
        elif boundary_type == "Conditional":
            boundary_function = self.apply_conditional_bcs
        else:
            raise Exception("Need to specify Dirichlet, Mixed or Neumann boundary conditions")

        uj = uold.clone()
        uuold = boundary_function(uold, g=g_old, kappa=self.kappa, q=q)
        
        if f == 0:
            f_new = torch.tensor(0)
        else:
            f_new = f(xx[1:-1, 1:-1], yy[1:-1, 1:-1], t + dt)

        D = torch.tensor([self.diag * dt + L])

        for j in range(nouter):

            uiplus = torch.zeros_like(uj) # set up jacobi iteration

            ui = uj.clone()

            uuj = boundary_function(uj, g=g_new, kappa=self.kappa, q=q)

            # Right hand side of Jacobi iteration - fixed for inner iterations
            b = L * uj + dt * f_new  + self.cmm(self.theta(uuold) - self.theta(uuj))

            for i in range(ninner):
                uui = boundary_function(ui, g=g_new, kappa=self.kappa, q=q) # apply dirichlet boundary condition

                div_k_grad_ui = 0.5 * (self.diff(self.kappa(uuj + self.zz) * uui)  \
                                            + self.kappa(uj + self.z) * self.diff(uui) \
                                            - ui * self.diff(self.kappa(uuj + self.zz))) # updating diffusion term
                
                Au = L * ui - dt * div_k_grad_ui

                uiplus = (ui  + (b - Au) / D).clone() # Jacobi iteration # omega = 1 does not work!!!
                #uiplus = jacobi2(Au, b, ui, D,  maxiterations=2) # These solvers give exactly the same thing

                assert torch.isinf(torch.sqrt(torch.mean((uiplus - ui)**2))) == False
                assert torch.isnan(torch.sqrt(torch.mean((uiplus - ui)**2))) == False

                if torch.sqrt(torch.mean((uiplus - ui)**2)) <= 1e-12:
                    print(f"Inner iterations converged after {i+1} iterations")

                    break

                if i == ninner - 1:
                    break

                ui = uiplus.clone()

            if torch.sqrt(torch.mean((uiplus - uj)**2)) <= 1e-12:
                print(f"Outer iterations converged after {j+1} iterations")
                break

            if j == nouter - 1:
                print(f"Outer iteration has not converged, error is {torch.sqrt(torch.mean((uiplus - uj)**2))}")
                break

            uj = uiplus.clone()        

        return uiplus    

class Richards_PC(Richards):
    def __init__(self, zz, dx, dy, theta, kappa, drain_l=0, drain_r=0):
         super().__init__(zz, dx, dy, theta, kappa, drain_l=0, drain_r=0)

    def forward(self, uold, t, dt, boundary_type, ninner=1, q=1, g=1, f=0, xx=1, yy=1):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
        L = 5 # lipschitz constant, should be 2.
 
        if boundary_type == "Dirichlet":
            boundary_function = self.apply_dirichlet_bcs
        elif boundary_type == "Mixed":
            boundary_function = self.apply_mixed_bcs
        elif boundary_type == "Neumann":
            boundary_function = self.apply_neumann_bcs
        elif boundary_type == "Conditional":
                    boundary_function = self.apply_conditional_bcs
        else:
            raise Exception("Need to specify Dirichlet, Mixed, Neumann or Conditional boundary conditions")
            
        uuold = boundary_function(uold, g=g, kappa=self.kappa, q=q)

        if f == 0:
            f_new = torch.tensor(0)
        else:
            f_new = f(xx[1:-1, 1:-1], yy[1:-1, 1:-1], t + dt)

        ujplus = torch.zeros_like(uold)
        uj = uold.clone()

        for i in range(ninner):
            # predictor step
            uuj = boundary_function(uj, g=g, kappa=self.kappa, q=q)
            # this is like a forward euler scheme
            div_k_grad_uj = 0.5 * (self.diff(self.kappa(uuj) * (uuj + self.zz)) \
                                     + self.kappa(uj) * self.diff(uuj + self.zz) \
                                          - (uj + self.z)* self.diff(self.kappa(uuj)))

            #div_k_grad_uj = self.diff(uuj)

            RHS = self.cmm(self.theta(uuold) - self.theta(uuj)) + dt * div_k_grad_uj + dt * f_new
            ujstar =  uj + 1 / L * RHS
            
            uujstar = boundary_function(ujstar, g=g, kappa=self.kappa, q=q)

            # corrector step
            ujpred = 0.5 * (uold + ujstar)
            uujpred = boundary_function(ujpred, g=g, kappa=self.kappa, q=q)

            div_k_grad_uj_pred = 0.5 * (self.diff(self.kappa(uujpred) * (uujpred + self.zz)) \
                                          + self.kappa(ujpred) * self.diff(uujpred + self.zz) \
                                            - (ujpred + self.z)* self.diff(self.kappa(uujpred)))

            #div_k_grad_uj_pred = self.diff(uujpred)

            RHS_pred = self.cmm(self.theta(uuold) - self.theta(uujstar)) + dt * div_k_grad_uj_pred
            ujplus = (uj + 1 / L * RHS_pred).clone()


            assert torch.isinf(torch.sqrt(torch.mean((ujplus - uj)**2))) == False
            assert torch.isnan(torch.sqrt(torch.mean((ujplus - uj)**2))) == False
            
            if torch.sqrt(torch.mean((ujplus - uj)**2)) <= 1e-12:
                print(f"Iterations converged after {i+1} iterations")
                break

            uj = ujplus.clone()

        return ujplus

        
class Richards_PC_MG(Richards):
    def __init__(self, zz, dx, dy, theta, kappa, drain_l=0, drain_r=0):
         super().__init__(zz, dx, dy, theta, kappa, drain_l=0, drain_r=0)

    def forward(self, uold, t, dt, boundary_type, ninner=1, q=1, g=1, f=0):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
        L = 5 # lipschitz constant, should be 2.
 
        if boundary_type == "Dirichlet":
            boundary_function = self.apply_dirichlet_bcs
        elif boundary_type == "Mixed":
            boundary_function = self.apply_mixed_bcs
        elif boundary_type == "Neumann":
            boundary_function = self.apply_neumann_bcs
        elif boundary_type == "Conditional":
                    boundary_function = self.apply_conditional_bcs
        else:
            raise Exception("Need to specify Dirichlet, Mixed, Neumann or Conditional boundary conditions")
            
        uuold = boundary_function(uold, g=g, kappa=self.kappa, q=q)

        if f == 0:
            f_new = torch.tensor(0)
        else:
            f_new = self.f(self.x[1:-1, 1:-1], self.y[1:-1, 1:-1], t + dt)

        ujplus = torch.zeros_like(uold)
        uj = uold.clone()


        ## First calculate the residual on each level
        iteration = 3
        nlevel = 3

        for MG in range(iteration):
            r = f

            r_s = []
            r_s.append(r)

            for i in range(1, nlevel):
                r = self.res(r)
                r_s.append(r)

            for i in reversed(range(1, nlevel)):
                n = 4



        ## Then perform a jacobi iteration on the coarsest mesh

        ## prolongate to next level

        ## move to coarsest level



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
