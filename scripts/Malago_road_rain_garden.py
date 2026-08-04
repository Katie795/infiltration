''' 
Infiltration of Malago Road rain garden using Richard's equation
Katie MacKenzie 
km2560@bath.ac.uk
'''
# # # ######################################## # # #
# # # #####   Import libraries and data  ##### # # #
# # # ######################################## # # #

import numpy as np 
import time 
import torch
import tomllib
import argparse
from AI4Flood.richards_solver import *
from AI4Flood.generate_csv_files import generate_rectangular_coords, generate_rectangular_padded_coords
from AI4Flood.richards_coefficients import theta_malago, kappa_malago

rainfall_data = np.genfromtxt("../data/Clifton_rain_modifed_for_rainfall_event.csv", delimiter=',')[1:, 0]
time_data = 60 * 15 * np.array(range(len(rainfall_data)))
def rainfall_flux(t):
    """Interpolated function for rainfall"""
    return dt * np.interp(t, time_data, rainfall_data) / (1000 * 15 * 60) #converting mm/15mins flux to m/second / timestep


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    action="store",
    help="name of the parameter file",
    default="config_rain_garden.toml"
)

args, _ = parser.parse_known_args()

with open(args.config, "rb") as f:
    config = tomllib.load(f)

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print(f"Running on device {device}")

# # # ##################################################### # # #
# # # #####   Numerical parameters from config file   ##### # # #
# # # ##################################################### # # #
dx       = torch.tensor([config["parameters"]["dx"]], device=device) 
nx       = torch.tensor([config["parameters"]["nx"]], device=device)
ny       = torch.tensor([config["parameters"]["ny"]], device=device)
length   = torch.tensor([config["parameters"]["length"]], device=device) 
depth    = torch.tensor([config["parameters"]["depth"]], device=device) 
dt       = torch.tensor([config["parameters"]["dt"]], device=device)
T        = torch.tensor([config["parameters"]["T"]], device=device)
save_fig = torch.tensor([config["parameters"]["save_fig"]], device=device)
n_out    = torch.tensor([config["parameters"]["n_out"]], device=device)

# TODO: write this as one compact function
generate_rectangular_coords(nx, nx, name="infiltration_", x_length=length, y_length=depth)
generate_rectangular_padded_coords(nx, nx, name="infiltration_padded_", x_length=length, y_length=depth)
x = torch.tensor(np.genfromtxt("../data/infiltration_x_coords.csv", delimiter=',')).double()#.cuda()
y = torch.tensor(np.genfromtxt("../data/infiltration_y_coords.csv", delimiter=',')).double()#.cuda()
x_padded = torch.tensor(np.genfromtxt("../data/infiltration_padded_x_coords.csv", delimiter=',')).double()#.cuda()
y_padded = torch.tensor(np.genfromtxt("../data/infiltration_padded_y_coords.csv", delimiter=',')).double()#.cuda()

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
diag = -np.array(w1.cpu())[0,0,1,1] # diagonal matrix

# # # ################################### # # #
# # # ######   Create field (tensor)  ### # # #
# # # ################################### # # #
print('============== Numerical parameters ===============')
print('Time step:', dt)
print('Final time:', T)
print('Initial time:', 0)
print('Diagonal component:', diag)


# # # ################################### # # #
# # # #######   Initialisation ########## # # #
# # # ################################### # # #

# TODO: better initial condition
u_old = 0.0 * torch.unsqueeze(torch.unsqueeze(torch.ones_like(x), 0), 0).to(torch.float64)#.cuda()
model = AI4RichardsPC(w1, wm, bias_initializer, y_padded, theta_malago, kappa_malago, diag=diag, f=0).to(device)

t = 0
i = 0
start = time.time()
print("starting timer")

with torch.no_grad():
    while t < T:
        u_new = model(u_old, t, dt, boundary_type="Neumann", ninner=2, q=rainfall_flux(t), dy=0.02)
        t += dt
        i += 1
        # save figure every n_out timesteps
        if save_fig == True and i % n_out == 0:
            np.savetxt(f"../results/u_{int(i/ 100)}.csv", u_new.cpu().detach().numpy()[0,0,:,:], delimiter=",")

        u_old = u_new.clone()

    end = time.time()
    print(f'Time to run simulation: {end-start}s')
