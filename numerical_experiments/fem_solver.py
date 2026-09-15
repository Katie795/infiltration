import numpy as np
from firedrake import *

def u_exact_FEM(x, y, t, V):
    s = Function(V)
    s.interpolate(x - y)
    left = -0.5 * s
    u_exact = project(left, V)
    right = -tan((exp(s) - 1) / (exp(s) + 1)) # this is written using UFL
    u_exact.assign(project(conditional(le(0, s), right, left), V))

    return u_exact

def kappa_FEM(u, V):
    base = Constant(2.0)
    kappa = project(base, V)
    kappa_left = 2.0 / (1.0 + u **2 ) # TODO: some confusion as to what this is
    kappa.assign(project(conditional(le(u, 0), kappa_left, base), V))
    return kappa

def theta_FEM(u, V):
    base = Constant(0.5 * pi**2)
    theta = project(base, V)
    atan_term = 2 * atan(u)**2
    theta.assign(project(conditional(le(u, 0), atan_term, base), V))
    return theta

def Richards_FEM(nx, nt, T, mesh, V, L=2, ninner=10, nouter=10):
    uj = Function(V)
    ui = Function(V)
    uold = Function(V)
    x, y = SpatialCoordinate(mesh)
    u_init = u_exact_FEM(x, y, 0, V)
    uj.interpolate(u_init)
    uold.interpolate(u_init)
    dt = T / nt
    t = 0
    while t <= T:
        for j in range(nouter):
            u = TrialFunction(V)
            v = TestFunction(V)
            # Right hand side of Jacobi iteration - fixed for inner iterations
            f = Function(V)
            f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
            #b = inner(L * uj, v) * dx #+  inner(theta_FEM(uold, V) - theta_FEM(uj, V), v) * dx
            b = inner(f, v) * dx
  
            #bc_function = u_exact_FEM(x, y, t, V)
            #bc = DirichletBC(V, bc_function, 1)
            
            #A = inner(L * ui, v) * dx  + dt * inner( kappa_FEM(uj, V) * grad(ui), grad(v)) * dx
            A = (inner(u, v)  + inner( grad(u), grad(v)) )* dx

            u = Function(V)
            solve(A == b, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}) # inner loop is done in here?
            
            if errornorm(u, u_exact_FEM(x, y, t, V)) <= 1e-12:
                print(f"Outer iterations converged after {j+1} iterations")
                break

            if j == nouter - 1:
                print(f"Outer iteration has not converged, error is {errornorm(u, u_exact_FEM(x, y, t, V))}")
                break

            uj.assign(u) 
        t += dt 
        uold.assign(u)    
    error = errornorm(u, u_exact_FEM(x, y, t, V))
    
    return u, error