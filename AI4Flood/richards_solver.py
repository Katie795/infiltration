import torch
import torch.nn as nn
from AI4Flood.operators import get_weights_linear_2D, construct_diff_filter

class Richards(nn.Module):
    def __init__(self, xx, zz, dx, dy, theta, kappa, n_MG, drain_l, drain_r):
        super().__init__()

        self.xx = xx
        self.zz = zz
        self.dx = dx
        self.dy = dy
        self.drain_l = drain_l
        self.drain_r = drain_r
        self.n_MG = n_MG

        self.theta = theta
        self.kappa = kappa

        [w_diff, wm1, wres, diag, bias] = get_weights_linear_2D(self.dx)
        
        self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        self.diff.weight.data = w_diff
        self.cmm.weight.data = wm1
        self.diag = diag

        self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0) # restriction
        self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest")) # prolongation
        self.res.weight.data = wres
        self.res.bias.data = bias
        self.diff.bias.data = bias
        self.cmm.bias.data = bias


        self.xx_grids = []
        self.zz_grids = []

        for i in range(self.n_MG - 1):
            xx, zz = self.restrict_grid(xx, zz)
            self.xx_grids.append(xx)
            self.zz_grids.append(zz)

    def find_closest_index(self, value, dx):

        return torch.round(value / dx).int()

    def restrict_grid(self, xx, yy):

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

    def apply_dirichlet_bcs(self, u, g_t=0, kappa=None, smoothing=False):
        """Applies a boundary condition g to a function u.
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function with u in interior cells and g on padded cells."""

        # adding padding to tensor
        u_new = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)
        
        if smoothing == True:
            return u_new

        u_new[0,0,0,:] = g_t[0,0,0,:]    # top dirichlet
        u_new[0,0,:,0] = g_t[0,0,:,0]    # left dirichlet
        u_new[0,0,:,-1] = g_t[0,0,:,-1]  # right dirichlet
        u_new[0,0,-1,:] = g_t[0,0,-1,:]  # bottom dirichlet

        return u_new
        
    def apply_mixed_bcs(self, u, g=1, kappa=1, q=2, smoothing=False):
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

        if smoothing == False:
            u_new[0,0,0,:] = g[0,0,0,:]                # top dirichlet
            u_new[0,0,-1, cut1:cut1] = u_new[0, 0, -2, cut1:cut1] - self.dy * (q / kappa(u_new[:, :, -2, cut1:cut1]) + 1)# bottom inhomogeneous neumann
        else:
            u_new[0,0,0,:] = 0 * g[0,0,0,:]                # top dirichlet
            u_new[0,0,-1, cut1:cut1] = u_new[0, 0, -2, cut1:cut1] 

        return u_new
    

    def apply_neumann_bcs(self, u, g_t=1, kappa=1, q=2, smoothing=False):
        """Applies a boundary condition g to a function u. Used when the soil is not saturated
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function"""

        # adding padding to tensor
        u_new = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)

        cut1 = self.find_closest_index(self.drain_l, self.dx)
        cut2 = self.find_closest_index(self.drain_r, self.dx)
        u_new[0,0,:,0]   = u_new[0,0,:,1]          # homogeneous neumann  
        u_new[0,0,:,-1]  = u_new[0,0,:,-2]         # homogeneous neumann
        u_new[0,0,-1, :cut1] = u_new[0,0,-2,:cut1] # homogeneous neumann
        u_new[0,0,-1, cut2:] = u_new[0,0,-2,cut2:] # homogeneous neumann

        id_active = u_new[0,0,-2, cut1:cut2] >= 0 
        id_inactive = u_new[0,0,-2, cut1:cut2] < 0

        u_new[0,0,-1, cut1:cut2][id_active] = 0 * u_new[0,0,-1, cut1:cut2][id_active] # set to zero!
        u_new[0,0,-1, cut1:cut2][id_inactive] = u_new[0,0,-2, cut1:cut2][id_inactive] # neumann bcs

        if smoothing == False:
            u_new[0,0,0,:]   = u_new[0,0,0,:] + self.dy * (q / kappa(u_new[:,:,0,:] ) + 1) # inhomogeneous neumann in m 
        else:
            u_new[0,0,0,:]   = u_new[0,0,0,:] 

        return u_new
    

    def apply_conditional_bcs(self, u, g_t=1, kappa=1, q=None, smoothing=False):
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


        u_new[0,0,:,0]   = u_new[0,0,:,1]          # homogeneous neumann  
        u_new[0,0,:,-1]  = u_new[0,0,:,-2]         # homogeneous neumann
        u_new[0,0,-1, :cut1] = u_new[0,0,-2,:cut1] + self.dx # homogeneous neumann
        u_new[0,0,-1, cut2:] = u_new[0,0,-2,cut2:] + self.dx # homogeneous neumann

        id_active = u_new[0,0,-2, cut1:cut2] >= 0 
        id_inactive = u_new[0,0,-2, cut1:cut2] < 0

        u_new[0,0,-1, cut1:cut2][id_active] = 0 * u_new[0,0,-1, cut1:cut2][id_active] # set to zero seepage occurs

        if smoothing == False:
             u_new[0,0,0,:][unsaturated] = u_new[0,0,0,:][unsaturated] + self.dy * (g_t[unsaturated] / kappa(u_new[:,:,0,:] )[0, 0, unsaturated] + 1) # inhomogeneous neumann in m 
             u_new[0,0,0,:][saturated] = g_t[saturated]
             u_new[0,0,-1, cut1:cut2][id_inactive] = u_new[0,0,-2, cut1:cut2][id_inactive] + self.dx 
        else:
            u_new[0,0,0,:][unsaturated] = u_new[0,0,0,:][unsaturated]
            u_new[0,0,0,:][saturated] = torch.zeros_like(u_new[0,0,0,:][saturated])
            u_new[0,0,-1, cut1:cut2][id_inactive] = u_new[0,0,-2, cut1:cut2][id_inactive] 

        return u_new    

    def A_L(self, uui, Kk, dt, diff, diff_k, uuold=1, L=1, zz=None):

        ui = uui[:, :, 1:-1, 1:-1]
        K = Kk[:, :, 1:-1, 1:-1]

        if zz == None:
            div_k_grad_ui = 0.5 * (diff(Kk * (uui))  \
                        + K * diff(uui) \
                        - (ui) * diff_k) # updating diffusion term
        else:
            z = zz[1:-1, 1:-1]
            div_k_grad_ui = 0.5 * (diff(Kk * (uui + zz))  \
                        + K * diff(uui + zz) \
                        - (ui + z) * diff_k) # updating diffusion term

        return L * ui - dt * div_k_grad_ui

    def A_L_corr(self, uui, Kk, dt, diff, diff_k, uuold=1, L=1, zz=None):
        """Here L=1 corresponds to the standard heat equation"""

        ui = uui[:, :, 1:-1, 1:-1]
        K = Kk[:, :, 1:-1, 1:-1]
        uold = uuold[:, :, 1:-1, 1:-1]

        if zz == None:
            div_k_grad_ui = 0.5 * (diff(Kk * (0.5 *(uui + uuold)))  \
                            + K * diff(0.5 *(uui + uuold)) \
                            - (0.5 *(ui + uold)) * diff_k) # updating diffusion term
        else:
            z = zz[1:-1, 1:-1]
            div_k_grad_ui = 0.5 * (diff(Kk * (0.5 *(uui + uuold) + zz))  \
                            + K * diff(0.5 *(uui + uuold) + zz) \
                            - (0.5 *(ui + uold) + z) * diff_k) # updating diffusion term

        return L * ui - dt * div_k_grad_ui

    def A_L_corr_res(self, uui, Kk, dt, diff, diff_k, L=1, zz=None):
        """Here L=1 corresponds to the standard heat equation"""

        ui = uui[:, :, 1:-1, 1:-1]
        K = Kk[:, :, 1:-1, 1:-1]

        if zz == None:
            div_k_grad_ui = 0.5 * (diff(Kk * (uui))  \
                        + K * diff(uui) \
                        - (ui) * diff_k) # updating diffusion term
        else:
            z = zz[1:-1, 1:-1]
            div_k_grad_ui = 0.5 * (diff(Kk* (uui + zz))  \
                        + K * diff(uui + zz) \
                        - (ui + z) * diff_k) # updating diffusion term

        return L * ui - 0.5 * dt * div_k_grad_ui


    @torch.no_grad()
    def V_cycle_MG(self, uj, b, t, dt, boundary_function, smoothing_iterations=3, MG_iterations=1, nlevel=1, phase="pred", uold=0, L=1, zz=None, g_old=None, g_list=None):
        dx = self.dx
        ui = uj.clone()
        g_org = g_old
        uuj_org = boundary_function(uj, g_t=g_org, kappa=self.kappa) # apply dirichlet boundary condition
        diff_diag = self.diag

        D_s = []
        dx_s = []
        Kk_s = []
        diff_filters = []
        diff_k = []

        dx_s.append(dx)
        Kk = self.kappa(uuj_org)
        Kk_s.append(Kk)
        diff_filters.append(construct_diff_filter(dx))
        diff_k.append(diff_filters[0](Kk))

        if phase == "pred":
            omega = 1
            uuold = 0
            A_fine = self.A_L
            A_coarse = self.A_L
        elif phase == "corr":
            omega = 0.5
            uuold = boundary_function(uold, g_t=g_old, kappa=self.kappa)
            A_fine = self.A_L_corr
            A_coarse = self.A_L_corr_res
        else:
            raise ValueError("Need to specify whether the phase is 'pred' or 'corr'")

        D = L + omega * diff_diag * dt * Kk[:,:,1:-1, 1:-1]
        D_s.append(D)

        for i in range(1, nlevel):
            # restricting to smaller mesh  
            uj = self.res(uj)
            g_t = g_list[i] if g_old is not None else None
            uuj = boundary_function(uj, g_t=g_t, kappa=self.kappa)
            Kk = self.kappa(uuj)
            dx = 2 * dx
            diff_filters.append(construct_diff_filter(dx))
            diff_diag = 0.25 * diff_diag
            D = L + omega * dt * diff_diag * Kk[:,:,1:-1,1:-1]

            D_s.append(D)
            dx_s.append(dx)
            Kk_s.append(Kk)
            diff_k.append(diff_filters[i](Kk))

        inv_Ds = [1 / D for D in D_s]
        zz_fine = zz[0] if zz is not None else None

        for MG in range(MG_iterations):
            uui = boundary_function(ui, g_t=g_org, kappa=self.kappa) # apply dirichlet boundary condition

            for j in range(smoothing_iterations): # Jacobi relaxation
                r = b - A_fine(uui, Kk_s[0], dt, diff=diff_filters[0], diff_k=diff_k[0], uuold=uuold, L=L, zz=zz_fine)
                ui = ui + inv_Ds[0] * 0.5 * r 
                uui = boundary_function(ui, g_t=g_org, kappa=self.kappa) # apply dirichlet boundary condition

            r_s = []
            r_s.append(r)
            
            for i in range(1, nlevel): # down each level
                r = self.res(r) # restrict to smaller level
                r_s.append(r)

            a = torch.zeros_like(r[:, :, 1:-1, 1:-1])

            for i in reversed(range(1, nlevel)):
                zz_coarse = zz[i] if zz is not None else None

                aa = boundary_function(a, smoothing=True) # APPLIES ZERO BOUNDARY CONDITIONS
                if i == nlevel-1: # if in coarsest grid
                    # Largest residual
                    a = r_s[i] * inv_Ds[i]
                else:   
                    for _ in range(smoothing_iterations):
                        a += (r_s[i] - A_coarse(aa, Kk_s[i], dt, diff=diff_filters[i], diff_k=diff_k[i], L=L, zz=zz_coarse)) * inv_Ds[i]
                        aa = boundary_function(a, smoothing=True)
            
                a = self.prol(a) 
            
            ui = ui + a
            uuinew = boundary_function(ui, g_t=g_org, kappa=self.kappa)

            # this is the same as sqrt(error)
            if torch.norm(uuinew - uui, p=2) <= 1e-6:
                break
            
            ui = ui + (b - A_fine(uuinew, Kk_s[0], dt, diff=diff_filters[0], diff_k=diff_k[0], uuold=uuold, L=L, zz=zz_fine)) * inv_Ds[0]

        return ui

class Richards_L(Richards):
    def __init__(self, xx, zz, dx, dy,  theta, kappa, n_MG, drain_l, drain_r):
         super().__init__(xx, zz, dx, dy, theta, kappa, n_MG, drain_l, drain_r)

    def forward(self, uold, t, dt, boundary, boundary_type, ninner=10, nouter=10, L=2, g_old=None, g_new=None, omega=1, g=0, f=0, zz=None):
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

        if zz is not None:
            z = self.zz[1:-1, 1:-1]

        tnew = t + dt

        uj = uold.clone()
        uuold = boundary_function(uold, g_t=g_old, kappa=self.kappa)
        
        if f == 0:
            f_new = torch.tensor(0)
        else:
            f_new = f(self.xx[1:-1, 1:-1], self.zz[1:-1, 1:-1], tnew)

        D = torch.tensor([self.diag * dt * torch.max(self.kappa(uold)) + L])

        for j in range(nouter):

            uiplus = torch.zeros_like(uj) # set up jacobi iteration
            ui = uj.clone()
            uuj = boundary_function(uj, g_t=g_new, kappa=self.kappa)

            # Right hand side of Jacobi iteration - fixed for inner iterations
            b = L * uj + dt * f_new  + self.cmm(self.theta(uuold) - self.theta(uuj))

            for i in range(ninner):
                uui = boundary_function(ui, g_t=g_new, kappa=self.kappa) # apply dirichlet boundary condition

                if zz == None:
                    div_k_grad_ui = 0.5 * (self.diff(self.kappa(uuj) * (uui))  \
                                                + self.kappa(uj) * self.diff(uui) \
                                                - ui  * self.diff(self.kappa(uuj))) # updating diffusion term
                else:
                    div_k_grad_ui = 0.5 * (self.diff(self.kappa(uuj) * (uui + self.zz))  \
                                                                    + self.kappa(uj) * self.diff(uui + self.zz) \
                                                                    - (ui + z) * self.diff(self.kappa(uuj))) # updating diffusion term
                
                Au = L * ui - dt * div_k_grad_ui

                uiplus = (ui  + (b - Au) / D).clone() # Jacobi iteration # omega = 1 does not work!!!            

                assert torch.isinf(torch.sqrt(torch.mean((uiplus - ui)**2))) == False
                assert torch.isnan(torch.sqrt(torch.mean((uiplus - ui)**2))) == False

                if torch.sqrt(torch.mean((uiplus - ui)**2)) <= 1e-6:
                    break

                if i == ninner - 1:
                    break

                ui = uiplus.clone()

            if torch.sqrt(torch.mean((uiplus - uj)**2)) <= 1e-5:
                print(f"Outer iterations converged after {j+1} iterations")
                break

            if j == nouter - 1:
                print(f"Outer iteration has not converged, error is {torch.sqrt(torch.mean((uiplus - uj)**2))}")
                break

            uj = uiplus.clone()        

        return uiplus    


class Richards_PC(Richards):
    def __init__(self, xx, zz, dx, dy, theta, kappa, n_MG, drain_l, drain_r):
        super().__init__(xx, zz, dx, dy, theta, kappa, n_MG, drain_l, drain_r)

    def forward(self, uold, t, dt, boundary_type, g=None, L=1, q=1, f=0, smoothing_iterations=3, MG_iterations=1, PC_iterations=1, nlevel=1, zz=None):
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
            
        tnew = t + dt

        if f == 0:
            f_old = torch.tensor(0)
            f_new = torch.tensor(0)

        else:
            f_old = f(self.xx[1:-1, 1:-1], self.zz[1:-1, 1:-1], torch.tensor(t))
            f_new = f(self.xx[1:-1, 1:-1], self.zz[1:-1, 1:-1], tnew)

        uj = uold.clone()

        # boundary conditions
        g_old = g(self.xx, self.zz, t) if g is not None else None
        if g is not None:
            g_list = []
            g_list.append(g(self.xx, self.zz, t))
            for i in range(self.n_MG - 1):
                g_list.append(g(self.xx_grids[i], self.zz_grids[i], tnew))
        else:
            g_list = None

        for i in range(PC_iterations):

            b = L * uj + dt * f_new  + self.theta(uold) - self.theta(uj)
            upred = self.V_cycle_MG(uj, b, t, dt, boundary_function, smoothing_iterations=smoothing_iterations, MG_iterations=MG_iterations,  nlevel=nlevel, phase="pred", L=L, zz=zz, g_old=g_old, g_list=g_list)

            b_half = L * upred + 0.5 * dt * (f_new + f_old) + self.theta(uold) - self.theta(upred)
            ucorr = self.V_cycle_MG(upred, b_half, t, dt, boundary_function, smoothing_iterations=smoothing_iterations, MG_iterations=MG_iterations,  nlevel=nlevel, phase="corr", uold=uold, L=L, zz=zz,  g_old=g_old, g_list=g_list)

            assert torch.isinf(torch.sqrt(torch.mean((upred - ucorr)**2))) == False
            assert torch.isnan(torch.sqrt(torch.mean((upred - ucorr)**2))) == False
            
            if torch.sqrt(torch.mean((uj - ucorr)**2)) <= 1e-5:
                print(f"Iterations converged after {i+1} iterations")
                break

            if i == PC_iterations - 1:
                print(f"Error is {torch.sqrt(torch.mean((ucorr - uj)**2))}")
                break

            uj = ucorr.clone()

        return ucorr
