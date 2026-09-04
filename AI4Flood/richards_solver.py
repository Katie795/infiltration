import torch
import torch.nn as nn
from AI4Flood.operators import get_weights_linear_2D
from AI4Flood.misc_functions import jacobi, jacobi2

class Richards(nn.Module):
    def __init__(self, zz, dx, dy, theta, kappa, xx, yy, drain_l=0, drain_r=0):
        super().__init__()

        self.zz = zz
        self.z = zz[1:-1, 1:-1]
        self.xx = xx
        self.yy = yy
        self.dx = dx
        self.dy = dy
        self.drain_l = drain_l
        self.drain_r = drain_r

        self.theta = theta
        self.kappa = kappa

        [w_diff, wm1, wA, wres, diag, bias] = get_weights_linear_2D(self.dx)
        
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.A    = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff.weight.data = w_diff
        self.cmm.weight.data = wm1
        self.A.weight.data = wA
        self.diag = diag

        self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0) # restriction
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest")) # prolongation
        self.res.weight.data = wres
        self.res.bias.data = bias
        self.diff.bias.data = bias
        self.cmm.bias.data = bias
        self.A.bias.data = bias

    def restrict_grid(self, xx, yy):

        x = xx[1:-1, 1:-1] # eliminating padding
        y = yy[1:-1, 1:-1]

        x_new = x[::2, ::2] # every second value
        y_new = y[::2, ::2]

        xx_new = torch.nn.functional.pad(x_new, (1,1,1,1), mode="constant", value=0)
        yy_new = torch.nn.functional.pad(y_new, (1,1,1,1), mode="constant", value=0)

        new_dx = 2 * (x[0, 1] - x[0, 0])
        new_dy = 2 * (y[0, 0] - y[1, 0])

        xx_new[1:-1, 0] = xx_new[1:-1, 1] - new_dx 
        xx_new[1:-1, -1] = xx_new[1:-1, -2] + new_dx 
        xx_new[0] = xx_new[1]
        xx_new[-1] = xx_new[-2] 

        yy_new[0, 1:-1] = yy_new[1, 1:-1] + new_dy
        yy_new[-1, 1:-1] = yy_new[-2, 1:-1] - new_dy
        yy_new[:, 0] = yy_new[:, 1]
        yy_new[:, -1] = yy_new[:, -2]
        
        return xx_new, yy_new

    def restrict_grid2(self, xx, yy):

        x = xx[1:-1, 1:-1] # eliminating padding
        y = yy[1:-1, 1:-1]
        x = x[None, None, :, :]
        y = y[None, None, :, :]

        x_new = self.res(x)# every second value
        y_new = self.res(y)

        xx_new = torch.nn.functional.pad(x_new, (1,1,1,1), mode="constant", value=0)[0,0]
        yy_new = torch.nn.functional.pad(y_new, (1,1,1,1), mode="constant", value=0)[0,0]

        new_dx = 2 * (x[0, 0, 0, 1] - x[0, 0, 0, 0])
        new_dy = 2 * (y[0, 0, 0, 0] - y[0, 0, 1, 0])

        xx_new[1:-1, 0] = xx_new[1:-1, 1] - new_dx 
        xx_new[1:-1, -1] = xx_new[1:-1, -2] + new_dx 
        xx_new[0] = xx_new[1]
        xx_new[-1] = xx_new[-2] 

        yy_new[0, 1:-1] = yy_new[1, 1:-1] + new_dy
        yy_new[-1, 1:-1] = yy_new[-2, 1:-1] - new_dy
        yy_new[:, 0] = yy_new[:, 1]
        yy_new[:, -1] = yy_new[:, -2]
        
        return xx_new, yy_new

    def apply_dirichlet_bcs(self, u, g=0, kappa=1, q=0, t=0, xx=0, yy=0):
        """Applies a boundary condition g to a function u.
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function with u in interior cells and g on padded cells."""

        # adding padding to tensor
        u_new = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)
        
        if isinstance(g, int):
            g_t = torch.zeros_like(u_new)
        else:
            g_t = g(xx, yy, t)

        u_new[0,0,0,:] = g_t[0,0,0,:]    # top dirichlet
        u_new[0,0,:,0] = g_t[0,0,:,0]    # left dirichlet
        u_new[0,0,:,-1] = g_t[0,0,:,-1]  # right dirichlet
        u_new[0,0,-1,:] = g_t[0,0,-1,:]  # bottom dirichlet

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

    def diff_nl(self, ui, uj, g, boundary_function, q=1, t=0, xx=0, yy=0, zz=0):

        if zz == 0:
            z = 0
        else:
            z = zz[1:-1, 1:-1]
        
        uui = boundary_function(ui, g=g, kappa=self.kappa, q=q, t=t, xx=xx, yy=yy) # apply dirichlet boundary condition
        uuj = boundary_function(uj, g=g, kappa=self.kappa, q=q, t=t, xx=xx, yy=yy) # apply dirichlet boundary condition

        div_k_grad_ui = 0.5 * (self.diff(self.kappa(uuj) * (uui + zz))  \
                + self.kappa(uj) * self.diff(uui + zz) \
                - (ui + z) * self.diff(self.kappa(uuj))) # updating diffusion term

        return div_k_grad_ui

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

    def F_cycle_MG(self, uj, b, D, g, t, dt, iteration, nlevel, boundary_function):
        ui = uj.clone()
        uj_org = uj.clone()
        tnew = t + dt
        xx = self.xx
        yy = self.yy

        u_s = []
        x_s = []
        y_s = []
        z_s = []
        x_s.append(xx)
        y_s.append(yy)
        u_s.append(uj_org)

        for i in range(1, nlevel):
            uj = self.res(uj)
            #zz = self.res(zz)
            xx, yy = self.restrict_grid2(xx, yy)

            u_s.append(uj)
            x_s.append(xx)
            y_s.append(yy)
            #zz = self.zz
            #z_s.append(zz) # TODO: this needs fixed as well

        for MG in range(iteration):
            r = b - self.diff_nl(ui, uj_org, g, boundary_function, xx=self.xx, yy=self.yy) * - dt   # PDE residual 
            r_s = []
            r_s.append(r)
            
            for i in range(1, nlevel): # down each
                r = self.res(r) # restrict to smaller level
                r_s.append(r)

            a = torch.zeros_like(xx[1:-1, 1:-1]) # place to store residuals
            a = a[None, None, :, :]

            for i in reversed(range(1, nlevel)):
                aa = boundary_function(a, g=g, t=tnew, xx=x_s[i], yy=x_s[i]) # APPLIES BOUNDARY CONDITIONS

                if i == nlevel-1: # if in coarsest grid
                    # Largest residual
                    a = r_s[i] / D
                else:
                    # updating the residual to include
                    # a = a + d
                    # where d is the solution to Ad = r_s[i]
                    # which is solved using a Jacobi iteration
                    a = a + (r_s[i] - self.diff_nl(a, u_s[i], g, boundary_function, xx=x_s[i], yy=x_s[i], zz=0) * - dt) / D 
                
                a = self.prol(a)   

            ui = ui + a
            ui = ui + 1 / D * (b - self.diff_nl(ui, uj_org, g, boundary_function, xx=self.xx, yy=self.yy)* -dt) 

        return ui

class Richards_L(Richards):
    def __init__(self, zz, dx, dy,  theta, kappa, xx, yy, drain_l=0, drain_r=0):
         super().__init__(zz, dx, dy, theta, kappa, xx, yy, drain_l=0, drain_r=0)

    def forward(self, uold, t, dt, boundary_type, ninner=10, nouter=10, L=2, omega=1, q=1, g_old=1, g_new=1, f=0):
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
            f_new = f(self.xx[1:-1, 1:-1], self.yy[1:-1, 1:-1], t + dt)

        D = torch.tensor([self.diag * dt * torch.max(self.kappa(uold)) + L])

        for j in range(nouter):

            uiplus = torch.zeros_like(uj) # set up jacobi iteration
            ui = uj.clone()
            uuj = boundary_function(uj, g=g_new, kappa=self.kappa, q=q)

            # Right hand side of Jacobi iteration - fixed for inner iterations
            b = L * uj + dt * f_new  + self.cmm(self.theta(uuold) - self.theta(uuj))

            for i in range(ninner):
                uui = boundary_function(ui, g=g_new, kappa=self.kappa, q=q) # apply dirichlet boundary condition

                div_k_grad_ui = 0.5 * (self.diff(self.kappa(uuj) * (uui + self.zz))  \
                                            + self.kappa(uj) * self.diff(uui + self.zz) \
                                            - (ui + self.z) * self.diff(self.kappa(uuj))) # updating diffusion term
                
                Au = L * ui - dt * div_k_grad_ui

                uiplus = (ui  + (b - Au) / D).clone() # Jacobi iteration # omega = 1 does not work!!!
                #uiplus = self.F_cycle_MG(Au, ui, b, D, iteration=1, nlevel=3)             

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
    def __init__(self, zz, dx, dy, theta, kappa, xx, yy, drain_l=0, drain_r=0):
         super().__init__(zz, dx, dy, theta, kappa, xx, yy, drain_l=0, drain_r=0)

    def forward(self, uold, t, dt, boundary_type, g_old=0, g_new=0, ninner=1, q=1, f=0, L=1):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
 
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
            
        uuold = boundary_function(uold, g=g_old, kappa=self.kappa, q=q)

        if f == 0:
            f_new = torch.tensor(0)
        else:
            f_new = f(self.xx[1:-1, 1:-1], self.yy[1:-1, 1:-1], t + dt)

        ujplus = torch.zeros_like(uold)
        uj = uold.clone()

        for i in range(ninner):
            # predictor step
            uuj = boundary_function(uj, g=g_new, kappa=self.kappa, q=q)
            # this is like a forward euler scheme
            div_k_grad_uj = 0.5 * (self.diff(self.kappa(uuj) * (uuj + self.zz)) \
                                     + self.kappa(uj) * self.diff(uuj + self.zz) \
                                          - (uj + self.z)* self.diff(self.kappa(uuj)))

            RHS = self.cmm(self.theta(uuold) - self.theta(uuj)) / dt + div_k_grad_uj + f_new
            ujstar =  uj + 1 / L *  RHS
            
            uujstar = boundary_function(ujstar, g=g_new, kappa=self.kappa, q=q)

            # corrector step
            ujpred = 0.5 * (uold + ujstar)
            uujpred = boundary_function(ujpred, g=g_new, kappa=self.kappa, q=q)

            div_k_grad_uj_pred = 0.5 * (self.diff(self.kappa(uujpred) * (uujpred + self.zz)) \
                                          + self.kappa(ujpred) * self.diff(uujpred + self.zz) \
                                            - (ujpred + self.z) * self.diff(self.kappa(uujpred )))

            RHS_pred = self.cmm(self.theta(uuold) - self.theta(uujstar))/ dt + div_k_grad_uj_pred + f_new
            ujplus = (uj + 1 / L * RHS_pred).clone()

            assert torch.isinf(torch.sqrt(torch.mean((ujplus - uj)**2))) == False
            assert torch.isnan(torch.sqrt(torch.mean((ujplus - uj)**2))) == False
            
            if torch.sqrt(torch.mean((ujplus - uj)**2)) <= 1e-12:
                print(f"Iterations converged after {i+1} iterations")
                break

            if i == ninner - 1:
                print(f"PC scheme has not converged, error is {torch.sqrt(torch.mean((ujplus - uj)**2))}")
                break

            uj = ujplus.clone()

        return ujplus



class Richards_PC2(Richards):
    def __init__(self, zz, dx, dy, theta, kappa, xx, yy, drain_l=0, drain_r=0):
        super().__init__(zz, dx, dy, theta, kappa, xx, yy, drain_l=0, drain_r=0)

    def forward(self, uold, t, dt, boundary_type, g=0, ninner=1, q=1, f=0, L=1):
        """Here we use a Jacobi iteration with an L method to solve richard's equation."""
 
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
            
        uuold = boundary_function(uold, g=g, kappa=self.kappa, q=q, t=t, xx=self.xx, yy=self.yy)
        tnew = t + dt

        if f == 0:
            f_new = torch.tensor(0)
        else:
            f_new = f(self.xx[1:-1, 1:-1], self.yy[1:-1, 1:-1], tnew)

        ujplus = torch.zeros_like(uold)
        uj = uold.clone()
        uuj = boundary_function(uj, g=g, kappa=self.kappa, q=q, t=tnew, xx=self.xx, yy=self.yy)

        b = dt * f_new  + self.cmm(self.theta(uuold) - self.theta(uuj))

        for i in range(ninner):
            # predictor step
            ujstar = self.F_cycle_MG(uj, b, self.diag, g, t, dt, 3, 3, boundary_function)
            uujstar = boundary_function(ujstar, g=g, kappa=self.kappa, q=q, t=tnew, xx=self.xx, yy=self.yy)

            # corrector step
            ujpred = 0.5 * (uold + ujstar)
            ujplus = self.F_cycle_MG(ujpred, b, self.diag, g, t, dt, 3, 3, boundary_function)

            assert torch.isinf(torch.sqrt(torch.mean((ujplus - uj)**2))) == False
            assert torch.isnan(torch.sqrt(torch.mean((ujplus - uj)**2))) == False
            
            if torch.sqrt(torch.mean((ujplus - uj)**2)) <= 1e-12:
                print(f"Iterations converged after {i+1} iterations")
                break

            if i == ninner - 1:
                print(f"PC scheme has not converged, error is {torch.sqrt(torch.mean((ujplus - uj)**2))}")
                break

            uj = ujplus.clone()

        return ujplus