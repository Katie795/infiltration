import numpy as np
from firedrake import *

def u_exact_FEM(x, y, t, V):
    s = Function(V)
    s.interpolate(x - y - t)
    left = -0.5 * s
    u_exact = Function(V)
    right = -tan(tanh(0.5 * s))
    u_exact.interpolate(conditional(le(s, 0), left, right), V)
    return u_exact

def kappa_FEM(u, V):
    base = Constant(2.0)
    kappa = Function(V)
    kappa_left = 2.0 / (1.0 + u **2 ) 
    kappa.interpolate(conditional(le(u, 0), kappa_left, base), V)
    return kappa

def theta_FEM(u, V):
    base = Constant(0.5 * pi**2)
    theta = Function(V)
    atan_term = 2 * atan(u)**2
    theta.interpolate(conditional(le(u, 0), base - atan_term, base), V)
    return theta

def Richards_FEM(dt, nt, mesh, V, L=2, ninner=10, nouter=10):
    uj = Function(V)
    uold = Function(V)
    x, y = SpatialCoordinate(mesh)
    u_init = u_exact_FEM(x, y, 0, V)
    uj.assign(u_init)
    uold.assign(u_init)
    t = 0
    for n in range(nt):
        t = (n + 1) * dt
        for j in range(nouter):
            u = TrialFunction(V)
            v = TestFunction(V)
            A = inner(L * u, v) * dx  + dt * inner(kappa_FEM(uj, V) * grad(u), grad(v)) * dx
            b = inner(L * uj, v) * dx +  inner(theta_FEM(uold, V) - theta_FEM(uj, V), v) * dx

            bc_function = u_exact_FEM(x, y, t, V)
            bc = DirichletBC(V, bc_function, "on_boundary")
 
            u_new = Function(V)
            solve(A == b, u_new, bcs=[bc], solver_parameters={'ksp_type': 'richardson',
                                                           'pc_type': 'jacobi',
                                                           'ksp_max_it': ninner,
                                                           'ksp_rtol': 1e-6}) 

            if errornorm(u_new, uj) <= 1e-6:
                print(f"Outer iterations converged after {j+1} iterations")
                uj.assign(u_new) 
                break

            if j == nouter - 1:
                print(f"Outer iteration has not converged, error is {errornorm(u_new, uj)}")
                uj.assign(u_new) 
                break

            nonlinear_error = errornorm(u_new, uj)

            uj.assign(u_new) 
        uold.assign(u_new)  
    u_final = uj  
    error = errornorm(u_final, u_exact_FEM(x, y, t, V))
    return u_final, error