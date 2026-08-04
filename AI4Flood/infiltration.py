### Module to simulate infiltration through soil
import numpy as np
import torch

indices = np.genfromtxt("data/rain_garden_indices.csv", delimiter=',')[2:, :2]
indices = indices.astype(np.int64)

input_shape = (1, 1, 302, 294)
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")

     
def apply_infiltration(values_h, values_h_initial, q):

    h_new = torch.maximum(values_h - q, values_h_initial) # make sure there are no negative values
    return h_new

def darcy(values_h, values_h_initial):
    """
    Apply Darcy's linear model to the fluid flow. 
    
    :param values_h: the height of the water from Bathymetry
    :param values_h_initial: the height of the ground from Bathymetry
    """

    kappa = 1e-10 # permeability of grainy sand (https://en.wikipedia.org/wiki/Permeability_(porous_media)#Ranges_of_common_intrinsic_permeabilities)
    mu = 1.1386 * 1e-6 # Kinematic viscosity (m^2/s) of water at 15 degrees Celsius (https://wiki.anton-paar.com/en/water/)
    L = 1.2 # Depth of the soil

    q_array2 = torch.zeros(input_shape, device=device)
    q_array2[0,0,301 - indices[:,1],indices[:,0]] = kappa / mu * ((values_h[0,0,301 - indices[:,1],indices[:,0]] - values_h_initial[0,0,301 - indices[:,1],indices[:,0]]) + L) / L * (0.2) ** 2
    
    h_new = apply_infiltration(values_h, values_h_initial, q_array2)

    return h_new
