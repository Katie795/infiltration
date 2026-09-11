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
import tomli as tomllib
import argparse
from AI4Flood.richards_solver import *
from AI4Flood.richards_coefficients import theta_malago, kappa_malago
from AI4Flood.generate_csv_files import generate_csv_files
from AI4Flood.misc_functions import clear_output_folder

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

rainfall_data = np.genfromtxt("../data/Clifton_rain_modifed_for_rainfall_event.csv", delimiter=',')[1:, 0]
time_data = 60 * 15 * np.array(range(len(rainfall_data)))

def rainfall_flux(t, dt):
    """Interpolated function for rainfall"""
    rainfall = np.interp(t, time_data, rainfall_data)
    return np.asarray(dt.cpu()) * rainfall / (1000 * 15 * 60) # converting mm/15mins flux to m/second / timestep

# Check if GPU is available 
device = torch.device(config["parameters"]["device"])
print(f"Running on device {device}")
clear_output_folder(config["parameters"]["results_folder"])
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

# # # ##################################################### # # #
# # # #####   Numerical parameters from config file   ##### # # #
# # # ##################################################### # # #
dx       = torch.tensor([config["parameters"]["dx"]]) 
dy       = torch.tensor([config["parameters"]["dy"]])
nx       = torch.tensor([config["parameters"]["nx"]])
ny       = torch.tensor([config["parameters"]["ny"]])
length   = torch.tensor([config["parameters"]["length"]]) 
depth    = torch.tensor([config["parameters"]["depth"]]) 
drain_l  = torch.tensor([config["parameters"]["drain_l"]])
drain_r  = torch.tensor([config["parameters"]["drain_r"]])
dt       = torch.tensor([config["parameters"]["dt"]])
T        = torch.tensor([config["parameters"]["T"]])
save_fig = torch.tensor([config["parameters"]["save_fig"]])
n_out    = torch.tensor([config["parameters"]["n_out"]])

x, y, xx, yy = generate_csv_files(nx, ny, length, depth, name="infiltration")

# # # ################################### # # #
# # # ######   Create field (tensor)  ### # # #
# # # ################################### # # #
print('============== Numerical parameters ===============')
print(f'Time step: {dt.item()} seconds')
print(f'Final time:  {T.item()} seconds = {T.item() / 3600} hours')

# # # ################################### # # #
# # # #######   Initialisation ########## # # #
# # # ################################### # # #

# assumption of some equilibrium
u_old = - torch.unsqueeze(torch.unsqueeze(y, 0), 0)
model = Richards_L(yy, dx, dy, theta_malago, kappa_malago, xx, yy, drain_l=drain_l, drain_r=drain_r)

t = 0
i = 0
start = time.time()
np.savetxt(f"../results/u_0.csv", u_old.cpu().detach().numpy()[0,0,:,:], delimiter=",")

with torch.no_grad():
    while t < T:
        u_new = model(u_old, t, dt, ninner=50, boundary_type="Neumann", L=2) 
        #u_new = model(u_old, t, dt, ninner=50, boundary_type="Neumann", L=2, iteration=10, nlevel=3)  # TODO: implement Neumann bcs in PC scheme!
        t += dt
        i += 1
        # save figure every n_out timesteps

        if save_fig == True and i % n_out == 0:
            np.savetxt(f"../results/u_{int(i/ n_out)}.csv", u_new.cpu().detach().numpy()[0,0,:,:], delimiter=",")

        u_old = u_new.clone()

    end = time.time()
    print(f'Time to run simulation: {end-start}s')
