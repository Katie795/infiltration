# Module containing the boundary conditions
# PLEASE IGNORE THIS PAGE, THIS WILL BE USED IN FUTURE DEVELOPMENT

import torch

class BoundaryCondition:
    def __init__(self,
                 dx=0,
                 dy=0,
                 g=0,
                 kappa=0,
                 q=0,
                 drain_l=0,
                 drain_r=0
                 ):
        
        self.dx = dx
        self.dy = dy
        self.drain_l = drain_l
        self.drain_r = drain_r
        self.g = g
        self.kappa = kappa
        self.q = q
        
    def find_closest_index(value, dx):
        return torch.round(value / dx).int()

    def apply_dirichlet_bcs(self, u, smoothing=False):
        """Applies a boundary condition g to a function u.
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function with u in interior cells and g on padded cells."""

        # adding padding to tensor
        u_new = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)
        
        if smoothing == True:
            return u_new

        u_new[0,0,0,:] = self.g[0,0,0,:]    # top dirichlet
        u_new[0,0,:,0] = self.g[0,0,:,0]    # left dirichlet
        u_new[0,0,:,-1] = self.g[0,0,:,-1]  # right dirichlet
        u_new[0,0,-1,:] = self.g[0,0,-1,:]  # bottom dirichlet

        return u_new
        
    def apply_mixed_bcs(self, u, smoothing=False):
        """Applies a boundary condition g to a function u.
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function"""

        # adding padding to tensor
        u_new = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)

        cut1 = self.find_closest_index(self.drain_l, self.dx)
        cut2 = self.find_closest_index(self.drain_r, self.dx)

        u_new[0,0,0,:] = self.g[0,0,0,:]                # top dirichlet
        u_new[0,0,:,0]       = u_new[0,0,:,1]      # left homogeneous neumann  
        u_new[0,0,:,-1]      = u_new[0,0,:,-2]     # right homogeneous neumann
        u_new[0,0,-1, :cut1] = u_new[0,0,-2,:cut1] # bottom left homogeneous neumann
        u_new[0,0,-1, cut2:] = u_new[0,0,-2,cut2:] # bottom right homogeneous neumann

        if smoothing == False:
            u_new[0,0,0,:] = self.g[0,0,0,:]                # top dirichlet
            u_new[0,0,-1, cut1:cut1] = u_new[0, 0, -2, cut1:cut1] - self.dy * (self.q / self.kappa(u_new[:, :, -2, cut1:cut1]) + 1)# bottom inhomogeneous neumann
        else:
            u_new[0,0,0,:] = 0 * self.g[0,0,0,:]                # top dirichlet
            u_new[0,0,-1, cut1:cut1] = u_new[0, 0, -2, cut1:cut1] 

        return u_new
    

    def apply_neumann_bcs(self, u, smoothing=False):
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
            u_new[0,0,0,:]   = u_new[0,0,0,:] + self.dy * (self.q / self.kappa(u_new[:,:,0,:] ) + 1) # inhomogeneous neumann in m 
        else:
            u_new[0,0,0,:]   = u_new[0,0,0,:] 

        return u_new
    

    def apply_conditional_bcs(self, u, smoothing=False):
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
             u_new[0,0,0,:][unsaturated] = u_new[0,0,0,:][unsaturated] + self.dy * (self.g[unsaturated] / self.kappa(u_new[:,:,0,:] )[0, 0, unsaturated] + 1) # inhomogeneous neumann in m 
             u_new[0,0,0,:][saturated] = self.g[saturated]
             u_new[0,0,-1, cut1:cut2][id_inactive] = u_new[0,0,-2, cut1:cut2][id_inactive] + self.dx 
        else:
            u_new[0,0,0,:][unsaturated] = u_new[0,0,0,:][unsaturated]
            u_new[0,0,0,:][saturated] = torch.zeros_like(u_new[0,0,0,:][saturated])
            u_new[0,0,-1, cut1:cut2][id_inactive] = u_new[0,0,-2, cut1:cut2][id_inactive] 

        return u_new
