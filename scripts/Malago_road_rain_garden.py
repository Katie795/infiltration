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
from AI4Flood.richards_coefficients import theta_malago, kappa_malago
from AI4Flood.generate_csv_files import generate_csv_files
from AI4Flood.misc_functions import clear_output_folder

clear_output_folder("/home/katie795/Flooding_workspace/Shallow_Water_Equations_NN4PDEs/results")
torch.set_default_dtype(torch.float64)

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
dy       = torch.tensor([config["parameters"]["dy"]], device=device)
nx       = torch.tensor([config["parameters"]["nx"]], device=device)
ny       = torch.tensor([config["parameters"]["ny"]], device=device)
length   = torch.tensor([config["parameters"]["length"]], device=device) 
depth    = torch.tensor([config["parameters"]["depth"]], device=device) 
drain_l  = torch.tensor([config["parameters"]["drain_l"]], device=device)
drain_r  = torch.tensor([config["parameters"]["drain_r"]], device=device)
dt       = torch.tensor([config["parameters"]["dt"]], device=device)
T        = torch.tensor([config["parameters"]["T"]], device=device)
save_fig = torch.tensor([config["parameters"]["save_fig"]], device=device)
n_out    = torch.tensor([config["parameters"]["n_out"]], device=device)

x, y, xx, yy = generate_csv_files(nx, ny, length, depth, name="infiltration")

# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #

bias_initializer = torch.tensor([0.0], device=device).double()
# Isotropic Laplacian  # TODO: make this vary in dy as well
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
print('Time step:', dt.data)
print('Final time:', T.data)
print('Initial time:', 0)
print('Diagonal component:', diag)

# # # ################################### # # #
# # # #######   Initialisation ########## # # #
# # # ################################### # # #

# assumption of some equilibrium
u_old = - torch.unsqueeze(torch.unsqueeze(y, 0), 0).to(torch.float64)
model = Richards_PC(w1, wm, bias_initializer, yy, dx, dy, theta_malago, kappa_malago, diag=diag, f=0, drain_l=drain_l, drain_r=drain_r).to(device)

t = 0
i = 0
start = time.time()
np.savetxt(f"../results/u_0.csv", u_old.cpu().detach().numpy()[0,0,:,:], delimiter=",")

with torch.no_grad():
    while t < T:
        u_new = model(u_old, t, dt, boundary_type="Neumann", ninner=2, q=rainfall_flux(t))
        t += dt
        i += 1
        # save figure every n_out timesteps

        if save_fig == True and i % n_out == 0:
            np.savetxt(f"../results/u_{int(i/ n_out)}.csv", u_new.cpu().detach().numpy()[0,0,:,:], delimiter=",")

        u_old = u_new.clone()

    end = time.time()
    print(f'Time to run simulation: {end-start}s')
