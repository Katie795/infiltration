''' Infiltration of Malago Road rain garden
Katie MacKenzie 
km2560@bath.ac.uk
'''
#-- Import general libraries
import numpy as np 
import time 
import torch
import tomllib
import argparse
from AI4Flood.richards_functions import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    action="store",
    help="name of the parameter file",
    default="config.toml"
)

args, _ = parser.parse_known_args()

with open(args.config, "rb") as f:
    config = tomllib.load(f)

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(f"Running on device {device}")

# # # ################################### # # #
# # # ######   Numerical parameters ###### # # #
# # # ################################### # # #
# create a torch constant

dt = torch.tensor([config["parameters"]["dt"]], device=device) # timestep (s)
dx = torch.tensor([config["parameters"]["dx"]], device=device) # spatial discretisation (m) (here it is 20cm)
#nx = #torch.tensor([config["parameters"]["nx"]], device=device) # number of points in x direction
ny = torch.tensor([config["parameters"]["ny"]], device=device) # number of points in y direction
length = 4   # 4m length
depth  = 1.2 # 1.2m depth
nx = 400 # 1cm gridspace
ny = 120 # 1cm gridspace
dt = 0.01
T = 10
save_fig = True

generate_rectangular_coords(nx, nx, name="infiltration_", x_length=length, y_length=depth)
generate_rectangular_padded_coords(nx, nx, name="infiltration_padded_", x_length=length, y_length=depth)
x = torch.tensor(np.genfromtxt("../data/infiltration_x_coords.csv", delimiter=',')).double()
y = torch.tensor(np.genfromtxt("../data/infiltration_y_coords.csv", delimiter=',')).double()
x_padded = torch.tensor(np.genfromtxt("../data/infiltration_padded_x_coords.csv", delimiter=',')).double()
y_padded = torch.tensor(np.genfromtxt("../data/infiltration_padded_y_coords.csv", delimiter=',')).double()

# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0], device=device).double()
# Isotropic Laplacian  
w1 = torch.tensor([[[[-1/3/dx**2], [-1/3/dx**2] , [-1/3/dx**2]],
                    [[-1/3/dx**2], [8/3/dx**2], [-1/3/dx**2]],
                    [[-1/3/dx**2], [-1/3/dx**2] , [-1/3/dx**2]]]], device=device).double()
# mass matrix
wm = torch.tensor([[[[0.0], [0.0], [0.0]],
                            [[0.0], [1.0], [0.0]],
                            [[0.0], [0.0], [0.0]]]], device=device).double()

w1 = torch.reshape(w1,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
wm = torch.reshape(wm,(1,1,3,3))

diag = -np.array(w1.cpu())[0,0,1,1]

# # # ################################### # # #
# # # ######   Create field (tensor)  ### # # #
# # # ################################### # # #

#######################################################
print('============== Numerical parameters ===============')
print('Time step:', dt)
print('Initial time:', 0)
print('Diagonal component:', diag)
#######################################################
# # # ################################### # # #
# # # #######   Initialisation ########## # # #
# # # ################################### # # #
input_shape = (1, 1, ny, nx)

def dirichlet_boundary_conditions(x, y, t=1):
    d_max = 2 
    d_min = 0
    x_max = torch.max(x)

    tensor = d_max * (x_max - 0.5 * x) / x_max
    return torch.unsqueeze(torch.unsqueeze(tensor, 0), 0).to(torch.float64)

plot_tensor(x_padded, y_padded, dirichlet_boundary_conditions(x_padded, y_padded)[0][0], "malago_dirichlet_boundary_conditions")

u_old = 0.01 * torch.unsqueeze(torch.unsqueeze(torch.ones_like(x), 0), 0).to(torch.float64)

bcs = apply_malago_bcs(u_old, dirichlet_boundary_conditions(x_padded, y_padded))
plot_tensor_infiltration(x_padded, y_padded, bcs[0][0], "malago_boundary_conditions")

model = AIRichardsPC(w1, wm, bias_initializer, x_padded, y_padded, theta_malago, kappa_malago, dirichlet_boundary_conditions, apply_malago_bcs, diag=diag).to(device)

t = 0
i = 0
start = time.time()
with torch.no_grad():

    while t < T:
        
        u_new = model(u_old, t, dt, ninner=50)
        t += dt
        i += 1

        if save_fig == True and i%10 == 0:
            #plot_tensor_infiltration(x, y, u_new[0][0], f"malago_{i}")
            np.savetxt(f"../results/u_{int(i/ 10)}.csv", u_new.cpu().detach().numpy()[0,0,:,:], delimiter=",")

        u_old = u_new.clone()

            
    end = time.time()
    print(f'Time to run simulation:{end-start}s')
