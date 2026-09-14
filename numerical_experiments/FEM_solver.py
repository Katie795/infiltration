import numpy as np
from firedrake import *

def u_exact_FEM(x, y, t):
    s = x - y - t
    left = -0.5 * s
    right = -tan((exp(s) - 1) / (exp(s) + 1)) # this is written using UFL
    ans = conditional(s < 0, left, right)
    return 

def kappa_FEM(u):
    base = 2.0
    kappa_left = 2.0 / (1.0 + u **2 ) # TODO: some confusion as to what this is
    ans = conditional(u < 0, kappa_left, base)
    return ans

def theta_FEM(u):
    u = u[0, 0]  # still assumes (1,1,n,n)

    base = 0.5 * torch.pi**2
    atan_term = 2 * torch.atan(u)**2

    ans = torch.where(u < 0, base - atan_term, base)

    return ans.unsqueeze(0).unsqueeze(0)

def Richards_FEM(u_init, nx, nt, T, f, g):

    mesh = UnitSquareMesh(nx, nx)
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = Function(V, name="solution_next")
    u_ = Function(V, name="solution")
    v = TestFunction(V)
    x = SpatialCoordinate(mesh)

    u_init = project(as_vector(u_exact_FEM(x[0], x[1], 0)))
    f = 0
    dt = T / nt
    bc_function = as_vector(u_exact_FEM(x[0], x[1], 0))
    bc0 = DirichletBC(V, bc_function)

    D = torch.tensor([self.diag * dt * torch.max(self.kappa(uold)) + L])

    t = 0
    while t <= T:
        for j in range(nouter):

            uiplus = torch.zeros_like(uj) # set up jacobi iteration
            ui = uj.clone()
            #uuj = boundary_function(uj, g=g, kappa=self.kappa, q=q, t=tnew, xx=self.xx, yy=self.yy)

            # Right hand side of Jacobi iteration - fixed for inner iterations
            b = inner(L * uj, v) * dx +  inner(theta(uuold) - theta(uuj), v) * dx

            for i in range(ninner):
                
                A = inner(L * ui, v) * dx  + dt * inner( kappa(uj) * grad(ui), grad(v)) * dx
                solve(a == L, ui, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'})     

                if sqrt(assemble(dot(u - f, u - f) * dx)) <= 1e-12:
                    print(f"Inner iterations converged after {i+1} iterations")
                    break

                if i == ninner - 1:
                    break

                ui = uiplus.clone()

            if sqrt(assemble(dot(u - f, u - f) * dx)) <= 1e-12:
                print(f"Outer iterations converged after {j+1} iterations")
                break

            if j == nouter - 1:
                print(f"Outer iteration has not converged, error is {sqrt(assemble(dot(u - f, u - f) * dx))}")
                break

            uj = uiplus 
        t += dt      

    return uiplus    