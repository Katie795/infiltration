# Module containing the boundary conditions
import torch


class BoundaryCondition:
    def __init__(self,
                 dx,
                 g = 0,
                 kappa = 0,
                 q = 0,
                 drain_l=0,
                 drain_r=0
                 ):
        
        self.dx = dx
        self.drain_l = drain_l
        self.drain_r = drain_r
        self.g = g
        self.kappa = kappa
        self.q = q
        
    def find_closest_index(value, dx):
        return torch.round(value / dx).int()

    def apply_dirichlet_bcs(self, u):
        """Applies a boundary condition g to a function u.
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function with u in interior cells and g on padded cells."""

        # adding padding to tensor
        uu = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)

        uu[0,0,0,:] = self.g[0,0,0,:]    # top dirichlet
        uu[0,0,:,0] = self.g[0,0,:,0]    # left dirichlet
        uu[0,0,:,-1] = self.g[0,0,:,-1]  # right dirichlet
        uu[0,0,-1,:] = self.g[0,0,-1,:]  # bottom dirichlet

        return uu
        
    def apply_mixed_bcs(self, u):
        """Applies a boundary condition g to a function u.
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function"""

        # adding padding to tensor
        uu = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)

        cut1 = self.find_closest_index(self.drain_l, self.dx)
        cut2 = self.find_closest_index(self.drain_r, self.dx)

        uu[0,0,0,:] = self.g[0,0,0,:]                # top dirichlet
        uu[0,0,:,0]       = uu[0,0,:,1]      # left homogeneous neumann  
        uu[0,0,:,-1]      = uu[0,0,:,-2]     # right homogeneous neumann
        uu[0,0,-1, :cut1] =uu[0,0,-2,:cut1] # bottom left homogeneous neumann
        uu[0,0,-1, cut2:] = uu[0,0,-2,cut2:] # bottom right homogeneous neumann
        uu[0,0,-1, cut1:cut1] = uu[0, 0, -2, cut1:cut1] - self.dy * (self.q / self.kappa(uu[:, :, -2, cut1:cut1]) + 1)# bottom inhomogeneous neumann

        return uu

    def apply_neumann_bcs(self, u):
        """Applies a boundary condition g to a function u. Used when the soil is not saturated
        
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: boundary function over nx+2 x ny+2 grid
        
        :returns uu: nx+2 x ny+2 function"""


       # adding padding to tensor
        uu = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)

        cut1 = self.find_closest_index(self.drain_l, self.dx)
        cut2 = self.find_closest_index(self.drain_r, self.dx)

        uu[0,0,0,:]   = uu[0,0,0,:] + self.dy * (self.q / self.kappa(uu[:,:,0,:] ) + 1) # inhomogeneous neumann in m 
        uu[0,0,:,0]   = uu[0,0,:,1]          # homogeneous neumann  
        uu[0,0,:,-1]  = uu[0,0,:,-2]         # homogeneous neumann
        uu[0,0,-1, :cut1] = uu[0,0,-2,:cut1] # homogeneous neumann
        uu[0,0,-1, cut2:] = uu[0,0,-2,cut2:] # homogeneous neumann

        id_active = uu[0,0,-2, cut1:cut2] >= 0 
        id_inactive = uu[0,0,-2, cut1:cut2] < 0

        uu[0,0,-1, cut1:cut2][id_active] = 0 * uu[0,0,-1, cut1:cut2][id_active] # set to zero!
        uu[0,0,-1, cut1:cut2][id_inactive] = uu[0,0,-2, cut1:cut2][id_inactive] # neumann bcs

        #uu[0,0,-1, cut1:cut2] = uu[0, 0, -2, cut1:cut2] - self.dy * (q / self.kappa(uu[:, :, -2, cut1:cut2]) + 1)# inhomogeneous neumann

        return uu

    def apply_conditional_bcs(self, u):
        """Applies conditional boundary conditions onto the soil 
                
        :args u: original function over nx x ny grid to which the boundary condition is applied
        :args g: here g is a nx + 2 tensor corresponding to the height of the water!!!
        
        :returns uu: nx+2 x ny+2 function"""


        # adding padding to tensor
        uu = torch.nn.functional.pad(u, (1,1,1,1), mode="constant", value=0)

        cut1 = self.find_closest_index(self.drain_l, self.dx)
        cut2 = self.find_closest_index(self.drain_r, self.dx)

        saturated   = uu[0,0, 1, :] >= 0 
        unsaturated = uu[0,0,1,:] < 0

        # TODO: does flux correspond to per unit space or per unit time?

        uu[0,0,0,:][unsaturated] = uu[0,0,0,:][unsaturated] + self.dy * (self.g[unsaturated] / self.kappa(uu[:,:,0,:] )[0, 0, unsaturated] + 1) # inhomogeneous neumann in m 
        uu[0,0,0,:][saturated] = self.g[saturated] 
        uu[0,0,:,0]   = uu[0,0,:,1]          # homogeneous neumann  
        uu[0,0,:,-1]  = uu[0,0,:,-2]         # homogeneous neumann
        uu[0,0,-1, :cut1] = uu[0,0,-2,:cut1] + self.dx # homogeneous neumann
        uu[0,0,-1, cut2:] = uu[0,0,-2,cut2:] + self.dx # homogeneous neumann

        id_active = uu[0,0,-2, cut1:cut2] >= 0 # why are these 
        id_inactive = uu[0,0,-2, cut1:cut2] < 0

        uu[0,0,-1, cut1:cut2][id_active] = 0 * uu[0,0,-1, cut1:cut2][id_active] # set to zero seepage occurs
        uu[0,0,-1, cut1:cut2][id_inactive] = uu[0,0,-2, cut1:cut2][id_inactive] + self.dx # neumann bcs - no seepage

        #uu[0,0,-1, cut1:cut2][id_active] = uu[0, 0, -2, cut1:cut2] - self.dy * (q / self.kappa(uu[:, :, -2, cut1:cut2]) + 1)# inhomogeneous neumann
        # This is not how seeoage boundary conditions are implemented!!
        return uu
