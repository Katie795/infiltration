import torch

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
    #u_new = u.clone()

    u_new = torch.zeros_like(g)
    u_new[0,0,1:-1, 1:-1] = u


    u_new[0,0,0,:] = g[0,0,0,:] # top
    u_new[0,0,:,0] = g[0,0,:,0] # left
    u_new[0,0,:,-1] = g[0,0,:,-1] # right
    u_new[0,0,-1,:] = g[0,0,-1,:]  # bottom

    return u_new.to(torch.float64)

def apply_malago_bcs(u, g, q=2):
    """Applies a boundary condition g to a function u.
    
    :args u: original function over nx x ny grid to which the boundary condition is applied
    :args g: boundary function over nx+2 x ny+2 grid
    
    :returns uu: nx+2 x ny+2 function"""

    u_new = torch.zeros_like(g)

    u_new[0,0,1:-1, 1:-1] = u

    cut1 = 177
    cut2 = 233

    u_new[0,0,0,:] = g[0,0,0,:]                # top dirichlet
    u_new[0,0,:,0]       = u_new[0,0,:,1]      # homogeneous neumann  
    u_new[0,0,:,-1]      = u_new[0,0,:,-2]     # homogeneous neumann
    u_new[0,0,-1, :cut1] = u_new[0,0,-2,:cut1] # homogeneous neumann
    u_new[0,0,-1, cut2:] = u_new[0,0,-1,cut2:] # homogeneous neumann
    u_new[0,0,-1, cut1:cut1] = u[0, 0, -1, cut1:cut1] + 0.01 * q # inhomogeneous neumann

    return u_new.to(torch.float64)