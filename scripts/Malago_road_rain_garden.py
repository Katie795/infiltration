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


# TODO tidy this up!!!
#dt = torch.tensor([config["parameters"]["dt"]], device=device) # timestep (s)
dx = 0.02 #  torch.tensor([config["parameters"]["dx"]], device=device) # spatial discretisation (m) (here it is 20cm)
#nx = #torch.tensor([config["parameters"]["nx"]], device=device) # number of points in x direction
ny = torch.tensor([config["parameters"]["ny"]], device=device) # number of points in y direction
length = 4   # 4m length
depth  = 1.2 # 1.2m depth
nx = 200 # 2cm #400 # 1cm gridspace
ny = 60 # 2cm #120 # 1cm gridspace
dt = 0.01
T = 21600 # 6 hours
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

def rainfall_flux(t):
    return dt * np.interp(t, time_data, rainfall_data) / (1000 * 15 * 60) #converting mm/15mins flux to m/second / timestep

u_old = 0.0 * torch.unsqueeze(torch.unsqueeze(torch.ones_like(x), 0), 0).to(torch.float64)
model = AIRichardsPC(w1, wm, bias_initializer, y, y_padded, theta_malago, kappa_malago, diag=diag).to(device)

t = 0
i = 0
start = time.time()
print("starting timer")

with torch.no_grad():

    while t < T:

        print(f"q is {rainfall_flux(t)}")

        
        u_new = model(u_old, t, dt, boundary_type="Neumann", ninner=2, q=rainfall_flux(t), dy=0.02)
        t += dt
        i += 1

        if save_fig == True and i % 100 == 0:
            #plot_tensor_infiltration(x, y, u_new[0][0], f"malago_{i}")
            np.savetxt(f"../results/u_{int(i/ 100)}.csv", u_new.cpu().detach().numpy()[0,0,:,:], delimiter=",")

        u_old = u_new.clone()

            
    end = time.time()
    print(f'Time to run simulation: {end-start}s')
