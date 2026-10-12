'''
 Amin Nadimy, Boyang Chen, Claire Heaney, Christopher Pain
 Department of Earth Science and Engineering
 Imperial College London

 modified by Katie MacKenzie 
 km2560@bath.ac.uk
'''
#-- Import general libraries
import numpy as np 
import time 
import torch
from AI4Flood.swe_solver import AI4SWE
import tomli as tomllib
import argparse
import AI4Flood.infiltration as infiltration 

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    action="store",
    help="name of the parameter file",
    default="config.toml"
)

parser.add_argument(
    "--infiltration",
    action="store_true",
    help="whether to apply infiltration to the rain garden"
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
nx = torch.tensor([config["parameters"]["nx"]], device=device) # number of points in x direction
ny = torch.tensor([config["parameters"]["ny"]], device=device) # number of points in y direction

# # # ################################### # # #
# # # ######    Linear Filter      ###### # # #
# # # ################################### # # #
bias_initializer = torch.tensor([0.0], device=device)
# Isotropic Laplacian  
# These are the weights for the convolutions for the neural network
# these are the same for uniform grids
# tensor is of shape (batchsize, nkernels, pixels in rows, pixels in columns)
# tensor is of shape (batchsize, channels out, rows, columns)
# These are related to finite element basis functions
w1 = torch.tensor([[[[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]],
                    [[1/3/dx**2], [-8/3/dx**2], [1/3/dx**2]],
                    [[1/3/dx**2], [1/3/dx**2] , [1/3/dx**2]]]], device=device)
# Gradient in x 
w2 = torch.tensor([[[[1/(12*dx)], [0.0], [-1/(12*dx)]],
                    [[1/(3*dx)] , [0.0], [-1/(3*dx)]] ,
                    [[1/(12*dx)], [0.0], [-1/(12*dx)]]]], device=device)
# Gradient in y 
w3 = torch.tensor([[[[-1/(12*dx)], [-1/(3*dx)], [-1/(12*dx)]],
                    [[0.0]       , [0.0]      , [0.0]]       ,
                    [[1/(12*dx)] , [1/(3*dx)] , [1/(12*dx)]]]], device=device)
# Consistant mass matrix 
wm = torch.tensor([[[[0.028], [0.11] , [0.028]],
                    [[0.11] ,  [0.44], [0.11]],
                    [[0.028], [0.11] , [0.028]]]], device=device)

w1 = torch.reshape(w1,(1,1,3,3)) # change shape from (1, 3, 3, 1) to (1, 1, 3, 3)
w2 = torch.reshape(w2,(1,1,3,3))
w3 = torch.reshape(w3,(1,1,3,3))
wm = torch.reshape(wm,(1,1,3,3))

#######################################################
################# Numerical parameters ################
ntime = config["parameters"]["nt"]    # Time steps
n_out = config["parameters"]["n_out"] # How often to save the results 
nrestart = 0                # Last time step for restart
ctime_old = 0               # Last ctime for restart
mgsolver = True             # Multigrid solver for non-hydrostatic pressure 
nsafe = 0.5                 # Continuty equation residuals
ctime = 0                   # Initialise ctime   
save_fig = True             # Save results
Restart = False             # Restart
eplsion_k = 1e-04           # Stablisation factor in Petrov-Galerkin for velocity
eplsion_eta = 1e-04         # Stablisation factor in Petrov-Galerkin for height
beta = 4                    # diagonal factor in mass term
real_time = torch.tensor([0.0], device=device)
istep = 0

# # # ################################### # # #
# # # ######   Physical parameters    ### # # #
# # # ################################### # # #
g_x = 0;g_y = 0;g_z = 9.81  # Gravity acceleration (m/s2) 
rho = 1/g_z                 # Resulting density
diag = -np.array(w1.cpu())[0,0,1,1]

# # # ################################### # # #
# # # ######   Create field (tensor)  ### # # #
# # # ################################### # # #
input_shape = (1, 1, ny, nx)
values_u     = torch.zeros(input_shape, device=device) # u
values_v     = torch.zeros(input_shape, device=device) # v
a_u          = torch.zeros(input_shape, device=device)
a_v          = torch.zeros(input_shape, device=device)
b_u          = torch.zeros(input_shape, device=device)
b_v          = torch.zeros(input_shape, device=device)
c_u          = torch.zeros(input_shape, device=device)
c_v          = torch.zeros(input_shape, device=device)
eta1         = torch.zeros(input_shape, device=device)
eta2         = torch.zeros(input_shape, device=device)
values_hh    = torch.zeros(input_shape, device=device)
dif_values_h = torch.zeros(input_shape, device=device)
values_h_old = torch.zeros(input_shape, device=device)
sigma_q      = torch.zeros(input_shape, device=device)
k_u          = torch.zeros(input_shape, device=device)
k_v          = torch.zeros(input_shape, device=device)
k_x          = torch.zeros(input_shape, device=device)
k_y          = torch.zeros(input_shape, device=device)
b            = torch.zeros(input_shape, device=device)
# Padding
input_shape_pd = (1, 1, ny + 2, nx + 2)
values_uu     = torch.zeros(input_shape_pd, device=device) # u boundary condition
values_vv     = torch.zeros(input_shape_pd, device=device) # v boundary condition
b_uu          = torch.zeros(input_shape_pd, device=device)
b_vv          = torch.zeros(input_shape_pd, device=device)
eta1_p        = torch.zeros(input_shape_pd, device=device)
dif_values_hh = torch.zeros(input_shape_pd, device=device)
values_hhp    = torch.zeros(input_shape_pd, device=device)
values_hp     = torch.zeros(input_shape_pd, device=device)
k_uu          = torch.zeros(input_shape_pd, device=device)
k_vv          = torch.zeros(input_shape_pd, device=device)
# stablisation factor
k1 = torch.ones(input_shape, device=device)*eplsion_eta
k2 = torch.zeros(input_shape, device=device)    
k3 = torch.ones(input_shape, device=device)*dx**2*0.5/dt # for transient problem
#######################################################
print('============== Numerical parameters ===============')
print('Mesh resolution:', values_v.shape)
print('Time step:', ntime)
print('Initial time:', ctime)
print('Diagonal component:', diag)
#######################################################
# # # ################################### # # #
# # # #######   Initialisation ########## # # #
# # # ################################### # # #

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
    
model = AI4SWE(w1, w2, w3, wm, bias_initializer, dx, beta, diag).to(device)

start = time.time()
with torch.no_grad():

    for itime in range(1, ntime+1):

        get_source(real_time)

        for t in range(2):
            [values_u, values_v, values_h, values_hh, dif_values_h, sigma_q] = model(values_u, values_uu, values_v, values_vv, values_H, values_h, 
                    values_hp, b_u, b_uu, b_v, b_vv, dt, rho, k1, k2, k3, eta1_p, source_h, dif_values_h, dif_values_hh, values_hh, values_hhp) # output          
        
        if args.infiltration:
            values_h = infiltration.darcy(values_h, values_h_initital)
            eta_filename = "eta_infiltration"
        else:
            eta_filename = "eta_impervious"
        
        real_time = real_time + dt
        istep +=1

        if np.max(np.abs(values_hh.cpu().detach().numpy())) > 10.0:
            print('Not converged !!!!!!')
            break

        print(f'Time step: {itime}, height correction: {np.max(values_hh.cpu().detach().numpy())}, time = {real_time.item():.2f}s')
        print(f"Maximum depth is {np.max((values_h - values_h_initital).cpu().detach().numpy())}m") 

        if save_fig == True and itime % n_out == 0:
            np.savetxt(f"results/{eta_filename}_{itime}.csv", (values_h - values_h_initital).cpu().detach().numpy()[0,0,:,:], delimiter=",")
            np.savetxt(f"results/h_road_{itime}.csv", values_h.cpu().detach().numpy()[0,0,:,:], delimiter=",")
            np.savetxt(f"results/u_{itime}.csv", (values_u).cpu().detach().numpy()[0,0,:,:], delimiter=",")
            np.savetxt(f"results/v_{itime}.csv", (values_v).cpu().detach().numpy()[0,0,:,:], delimiter=",")
            
    end = time.time()
    print(f'Time to run simulation:{end-start}s')
