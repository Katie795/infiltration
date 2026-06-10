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
from AI4Flood.richards_functions import generate_rectangular_coords, generate_rectangular_padded_coords, theta_malago, kappa_malago, AIRichards


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
T = 1
save_fig = True

generate_rectangular_coords(nx, nx, name="infiltration_", x_length=length, y_length=depth)
generate_rectangular_padded_coords(nx, nx, name="infiltration_padded_", x_length=length, y_length=depth)
x = torch.tensor(np.genfromtxt("../data/infiltration_x_coords.csv", delimiter=',')).double()
y = torch.tensor(np.genfromtxt("../data/infiltration_y_coords.csv", delimiter=',')).double()
x_padded = torch.tensor(np.genfromtxt("../data/infiltration_padded_x_coords.csv", delimiter=',')).double()
y_padded = torch.tensor(np.genfromtxt("../data/infiltration_y_coords.csv", delimiter=',')).double()


# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0], device=device)
# Isotropic Laplacian  
w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                    [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                    [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)
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
values_h = torch.zeros(input_shape, device=device)
values_H = torch.zeros(input_shape, device=device)

with open('data/full_domain.csv', 'r') as file:
    mesh = np.genfromtxt(file, delimiter=',')

mesh = mesh.reshape(ny,nx)
values_H[0,0,:,:] = torch.tensor(mesh, device=device)

# defining the source term
source_h = torch.zeros(input_shape, device=device)

# Transfer array into tensor
values_h = values_H
values_h_initital = values_H
values_H = - values_H ## this is used for info about the bathymetry of the surface! Values_H is fixed!

def get_source(time):
    # here time is in seconds
    if time < 100:
        # 10mm / hour ( heavy rain )
        # 10 / (60 * 60) = 0.00278 m / second
        source_h[0,0, :, :] = 0.00278 * dt # source times timestep
    elif time >= 100:
        source_h[0,0, :, :] = 0
    
model = AIRichards(w1, wm, bias_initializer, x_padded, y_padded, theta_malago, kappa_malago, boundary_conditions, diag=diag).to(device)

t = 0
i = 0
start = time.time()
with torch.no_grad():

    while t < T:
        
        u_new = model(u_old, t, dt, ninner=50, nouter=50)
        t += dt
        i += 1

        if save_fig == True:
            np.savetxt(f"results/u_{i}.csv", u_new.cpu().detach().numpy()[0,0,:,:], delimiter=",")

            
    end = time.time()
    print(f'Time to run simulation:{end-start}s')
