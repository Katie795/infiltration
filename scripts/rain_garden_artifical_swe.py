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

rainfall_data = np.genfromtxt(config["IO"]["rainfall_data"], delimiter=',')[1:, 0]
time_data = 60 * 15 * np.array(range(len(rainfall_data))) # TODO need to make this variable with any input data

device = torch.device(config["processor"]["device"])
print(f"Running on device {device}")
clear_output_folder("../results")
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)

# # # ##################################################### # # #
# # # #####   Numerical parameters from config file   ##### # # #
# # # ##################################################### # # #

length   = torch.tensor([config["geometry"]["length"]]) 
depth    = torch.tensor([config["geometry"]["depth"]]) 
drain_l  = torch.tensor([config["geometry"]["drain_l"]])
drain_r  = torch.tensor([config["geometry"]["drain_r"]])
dx       = torch.tensor([config["mesh"]["dx"]]) 
dy       = torch.tensor([config["mesh"]["dy"]])
nx       = torch.tensor([config["mesh"]["nx"]])
ny       = torch.tensor([config["mesh"]["ny"]])
dt       = torch.tensor([config["time"]["dt"]])
T        = torch.tensor([config["time"]["T"]])
save_fig = torch.tensor([config["IO"]["save_fig"]])
n_out    = torch.tensor([config["IO"]["n_out"]])

x, y, xx, yy = generate_csv_files(nx, ny, length, depth, name="infiltration")

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
        g = 5 * t * dt * (4 + dx - xx[0])
        
        u_new = model(u_old, t, dt, boundary_type="Conditional", ninner=2, q=0, g=g)
        t += dt
        i += 1
        # save figure every n_out timesteps

        if save_fig == True and i % n_out == 0:
            np.savetxt(f"../results/u_{int(i/ n_out)}.csv", u_new.cpu().detach().numpy()[0,0,:,:], delimiter=",")

        u_old = u_new.clone()

    end = time.time()
    print(f'Time to run simulation: {end-start}s')
