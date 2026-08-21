import os, shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

def clear_output_folder(folder):

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return

def plot_tensor(x, y, tensor, name):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    height = ax.pcolormesh(x, y, tensor)
    cb = plt.colorbar(height, label=name, ax=ax)
    plt.title(name)
    plt.savefig(f"../plots/{name}")
    plt.close()

def plot_tensor_infiltration(x, y, tensor, name):
    fig, ax = plt.subplots(1,1, figsize=(16,4.8))
    levels = MaxNLocator(nbins=15).tick_values(tensor.min(), tensor.max())
    cmap = plt.colormaps['PiYG']
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    height = ax.pcolormesh(x, y, tensor, cmap=cmap)
    cb = plt.colorbar(height, label=name, ax=ax)
    plt.title(name)
    plt.savefig(f"../plots/{name}")

def calculate_dx(u, dx):

    u_dx = (u[:, 1:] - u[:, :-1]) / dx
    return u_dx[:-1] # to remain same shape as dy

def calculate_dy(u, dy):
    u_dy = (u[:-1] - u[1:]) / dy
    return u_dy[:, :-1] # to remain same shape as dx

def calculate_grad(u, dx, dy):
    u_dx = calculate_dx(u, dx)
    u_dy = calculate_dy(u, dy)
    return u_dx, u_dy



def jacobi(A, b, x_old, maxiterations=100, tol=1e-12):

    A = torch.squeeze(A)
    b = torch.squeeze(b)
    x = torch.squeeze(x_old)



    assert dd(A), "A is not diagonally dominant, a requirement for jacobi to succeed."

    D = torch.diag(A)
    D_inv = torch.diagflat(1/D)

    for _ in range(maxiterations):
        dx = torch.matmul(D_inv, (b - torch.matmul(A, x)))
        x = x + dx
        if torch.max(dx) <= tol:
            break
    x = torch.reshape(x, x_old.shape)
    return x

def jacobi2(Ax, b_old, x, D, maxiterations=100, tol=1e-12):

    Ax = torch.squeeze(Ax)
    b = torch.squeeze(b_old)

    #assert dd(A), "A is not diagonally dominant, a requirement for jacobi to succeed."

    #D = torch.diag(A)
    D_inv = 1 / D

    for _ in range(maxiterations):
        dx = D_inv * (b - Ax)
        x = x + dx
        if torch.max(dx) <= tol:
            break
    x = torch.reshape(x, b_old.shape)

    # OF COURSE IT DOESNT WORK
    
  
    return x

def conjugate_gradient(Ax, b_old, x, maxiterations=100, tol=1e-12):

    L = Ax
    
    Ax = torch.squeeze(Ax)
    b = torch.squeeze(b_old)
    r = b - Ax
    p = r

    for _ in range(maxiterations):
        dx = D_inv * (b - Ax)
        x = x + dx
        if torch.max(dx) <= tol:
            break
    x = torch.reshape(x, b_old.shape)


def gauss_seidel():
    return

def dd(A):
    """Checks whether the matrix is diagonally dominant"""
    D = torch.diag(A)
    S = torch.sum(torch.abs(A), axis=1) - D
    if torch.all(D > S):
        return True
    else:
        return False