"""Getting a simple richards example working, comparing to Horung-Messing problem."""
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from richards_functions import *

# h = 0.1 should give an L2 error of 0.01
nx_array = torch.tensor([10, 20, 40, 80])

length = 10 # domain is [0, 10] x [0, 10]

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if is_gpu else "cpu")
print(f"Running on device {device}")

loss = nn.MSELoss()

bias_initializer = torch.tensor([0.0], device=device)
beta = 4                    # diagonal factor in mass term

t = 0

L2_error_dx = []
L2_error_dt = []
niterations = []

#dt = 0.01
T = 0.01

with torch.no_grad():
    for nx in nx_array:
        t=0
        dx = length / nx
        dt = 0.0005
        #dt = 0.01 * dx ** 2
        print(f"dt is {dt}")

        generate_coords(nx, nx, name="horung_", length=length)
        generate_padded_coords(nx, nx, name="horung_padded_", length=length)
        x = torch.tensor(np.genfromtxt("data/horung_x_coords.csv", delimiter=','))
        y = torch.tensor(np.genfromtxt("data/horung_y_coords.csv", delimiter=','))
        x_padded = torch.tensor(np.genfromtxt("data/horung_padded_x_coords.csv", delimiter=','))
        y_padded = torch.tensor(np.genfromtxt("data/horung_padded_y_coords.csv", delimiter=','))

        u_init = u_exact(x,y,0) # initial condition
        uj   = u_init.clone()   # 
        uold = u_init.clone()

        w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                    [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                    [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)
        
        # Consistant mass matrix 
        wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
                            [[0.11] ,  [0.44], [0.11]],
                            [[0.028], [0.11] , [0.028]]]], device=device)

        w1 = torch.reshape(w1,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
        wm = torch.reshape(wm,(1,1,3,3))

        diag = -8/3/dx**2

        model = AIRichards(w1, wm, bias_initializer, beta, diag, length=length, nx=nx).to(device)

        while t < T:
            u_new, nit = model(x_padded, y_padded, uold, uj, t, dt, imax=10000)
            t += dt
            print(f"t = {t}")
            if t >= T:
                print("t has reached the final time!")
                break
            uold = u_new.clone()

        error = u_new - u_exact(x, y, t) # spatial error

        plot_tensor(x_padded, y_padded, u_exact(x_padded,y_padded, dt)[0,0,:,:].cpu().numpy(), "exact_solution")
        plot_tensor(x, y, u_new[0,0,:,:].cpu().numpy(), "model_solution")
        plot_tensor(x, y, torch.abs(error)[0,0,:,:].cpu().numpy(), "model_exact_difference")

        L2_error_dx.append(loss(u_new, u_exact(x, y, t)).cpu().numpy())
        niterations.append(nit)

### convergence plot ###
print(f"L2_error is {np.array(L2_error_dx)}")
fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.loglog(1/(nx_array.cpu().numpy()), L2_error_dx, linestyle="dashed", marker="o", label=rf"$\Delta t = {dt}$")
plt.plot(1/(nx_array.cpu().numpy()), 0.001 *(1/(nx_array.cpu().numpy()))**2, color="black", linestyle="dotted", label=r"Slope $= -2$" )
ax.set_xlabel(r"$\Delta x$", fontsize=12)
ax.set_ylabel(r"$L^2$ error", fontsize=12)
plt.title(f"Error against mesh-size, total time T = {T}", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"plots/mesh_error")

data = {"nx": nx_array.cpu().numpy(), "L2 error": np.array(L2_error_dx), "N iterations": np.array(niterations)}
df = pd.DataFrame(data=data)

print(df.to_latex(index=False))

'''
dt = 0.01
t = 0
T = 0.1

dt_array = np.array([0.1, 0.05, 0.02])

with torch.no_grad():
    nx = 40
    dx = length / nx

    generate_coords(nx, nx, name="horung_", length=length)
    generate_padded_coords(nx, nx, name="horung_padded_", length=length)
    x = torch.tensor(np.genfromtxt("data/horung_x_coords.csv", delimiter=','))
    y = torch.tensor(np.genfromtxt("data/horung_y_coords.csv", delimiter=','))
    x_padded = torch.tensor(np.genfromtxt("data/horung_padded_x_coords.csv", delimiter=','))
    y_padded = torch.tensor(np.genfromtxt("data/horung_padded_y_coords.csv", delimiter=','))

    # set up initial tensors
    u_init = u_exact(x,y,0)
    input_shape = (1, 1, nx + 1, nx + 1)
    input_shape_pd = (1, 1, nx + 3, nx + 3)
    uj = torch.zeros(input_shape, device=device)
    uold = torch.zeros(input_shape, device=device)
    uj.data.copy_(u_init.data)
    uold.data.copy_(u_init.data)


    w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)

    # Consistant mass matrix 
    wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
                        [[0.11] ,  [0.44], [0.11]],
                        [[0.028], [0.11] , [0.028]]]], device=device)

    #wm = torch.tensor([[[[dx**2/36], [dx**2/9] , [dx**2/36]],
    #                    [[dx**2/9] ,  [dx**2*4/9], [dx**2/9]],
    #                    [[dx**2/36], [dx**2/9] , [dx**2/36]]]], device=device)

    w1 = torch.reshape(w1,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
    wm = torch.reshape(wm,(1,1,3,3))
    diag = -np.array(w1.cpu())[0,0,1,1]

    model = AIRichards(w1, wm, bias_initializer, beta, diag, length=length, nx=nx, omega=0.1).to(device)

    for dt in dt_array:
        t=0
        while t <= T:
            unew = model(x_padded, y_padded, uold, uj, t, dt, imax=1000)
            t += dt
            print(f"t = {t}")
            if t == T:
                print("t has reached the final time!")
                break
            uold.data.copy_(unew)

        error = unew - u_exact(x, y, T)

        plot_tensor(x_padded, y_padded, u_exact(x_padded,y_padded, t+dt)[0,0,:,:].cpu().numpy(), "exact_solution")
        plot_tensor(x, y, unew[0,0,:,:].cpu().numpy(), "model_solution")
        plot_tensor(x, y, torch.abs(error)[0,0,:,:].cpu().numpy(), "model_exact_difference")

        L2_error_dt.append(loss(unew, u_exact(x, y, T)).cpu().numpy())

### convergence plot ###
print(f"L2_error is {L2_error_dt}")

fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.loglog(dt_array, L2_error_dt, linestyle="dashed", marker="o", label=r"$\Delta x = 0.5$")
plt.plot(dt_array, 0.001 *(dt_array), color="black", linestyle="dotted", label=r"Slope $= -1$" )
ax.set_xlabel(r"$\Delta t$", fontsize=18)
ax.set_ylabel(r"$L^2$ error", fontsize=18)
plt.title("Error against time-step", fontsize=22)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig(f"plots/time_error")
'''