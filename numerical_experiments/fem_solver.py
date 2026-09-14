import numpy as np
from firedrake import *

def u_exact_FEM(x, y, t, V):
    s = x - y - t
    left = -0.5 * s
    right = -tan((exp(s) - 1) / (exp(s) + 1)) # this is written using UFL
    ans = conditional(s < 0, left, right)

    print(type(ans))
    return 

def kappa_FEM(u, V):
    base = 2.0
    kappa = project(base, V)
    kappa_left = 2.0 / (1.0 + u **2 ) # TODO: some confusion as to what this is
    ans = conditional(u < 0, kappa_left, base)
    kappa.assign(project(conditional(le(u, 0), kappa_left, base)))

    return kappa

def theta_FEM(u, V):
    base = 0.5 * pi**2
    atan_term = 2 * arctan(u)**2

    ans = conditional(u < 0, base - atan_term, base)

    return ans

def Richards_FEM(nx, nt, T):

    mesh = UnitSquareMesh(nx, nx)
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = Function(V, name="solution_next")
    uj = Function(V)
    ui = Function(V)
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)

    u_init = project(as_vector(u_exact_FEM(x[0], x[1], 0)))
    uj.assign(u_init)
    dt = T / nt
    bc_function = as_vector(u_exact_FEM(x[0], x[1], 0))
    bc0 = DirichletBC(V, bc_function)

    D = torch.tensor([self.diag * dt * torch.max(self.kappa(uold)) + L])

    t = 0
    while t <= T:
        for j in range(nouter):
            # Right hand side of Jacobi iteration - fixed for inner iterations
            b = inner(L * uj, v) * dx +  inner(theta(uuold) - theta(uuj), v) * dx

            for i in range(ninner):
                bc_function = as_vector(u_exact_FEM(x[0], x[1], t))
                bc = DirichletBC(V, bc_function)
                
                A = inner(L * ui, v) * dx  + dt * inner( kappa(uj) * grad(ui), grad(v)) * dx
                solve(a == L, ui, bcs=[bc], solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'}) 

                if sqrt(assemble(dot(u - f, u - f) * dx)) <= 1e-12:
                    print(f"Inner iterations converged after {i+1} iterations")
                    break

                if i == ninner - 1:
                    break


            if sqrt(assemble(dot(u - f, u - f) * dx)) <= 1e-12:
                print(f"Outer iterations converged after {j+1} iterations")
                break

            if j == nouter - 1:
                print(f"Outer iteration has not converged, error is {sqrt(assemble(dot(u - f, u - f) * dx))}")
                break

            uj.assign(ui) 
        t += dt     
    error = sqrt(assemble(dot(u_new - u_exact_FEM(x, y, t), u_new - u_exact_FEM(x, y, t))))
    
    return ui, error