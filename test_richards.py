import numpy as np
import torch
from richards_functions import *

is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")


###
# Dirichlet BCs test
# Kappa and theta test
###

nx = 10
length = 10
t=0

generate_coords(nx, nx, name="horung_", length=length)
generate_padded_coords(nx, nx, name="horung_padded_", length=length)
x = torch.tensor(np.genfromtxt("data/horung_x_coords.csv", delimiter=','))
y = torch.tensor(np.genfromtxt("data/horung_y_coords.csv", delimiter=','))
x_padded = torch.tensor(np.genfromtxt("data/horung_padded_x_coords.csv", delimiter=','))
y_padded = torch.tensor(np.genfromtxt("data/horung_padded_y_coords.csv", delimiter=','))


def test_coordinates():
    assert torch.equal(x, x_padded[1:-1, 1:-1])

def test_boundaries():

    u_ex = u_exact(x_padded,y_padded,t)[0,0,1:nx+2,1:nx+2]
    uu = dirichlet_boundary_condition(u_exact(x,y,t), u_exact(x_padded,y_padded,t))

    assert torch.equal(u_ex, uu[0,0,1:nx+2,1:nx+2])

def test_kappa():
    input1 = torch.tensor([[[[-0.5]]]])
    input2 = torch.tensor([[[[0.5]]]])
    input3 = torch.tensor([[[[-0.25]]]])
    input4 = torch.tensor([[[[0.25]]]])

    assert Kappa(input1) == 1.6
    assert Kappa(input2) == 2
    assert Kappa(input3) == 1.88235294118
    assert Kappa(input4) == 2


def test_theta():
    input1 = torch.tensor([[[[-0.5]]]], device=device)
    input2 = torch.tensor([[[[0.5]]]], device=device)
    input3 = torch.tensor([[[[-0.25]]]])
    input4 = torch.tensor([[[[0.25]]]])

    assert torch.isclose(theta(input1), torch.tensor(0.5 * torch.pi**2, device=device) - 2 * torch.arctan(torch.tensor(-0.5, device=device))**2)
    assert torch.isclose(theta(input2), torch.tensor(0.5 * torch.pi**2, device=device))
    assert torch.isclose(theta(input3), torch.tensor(0.5 * torch.pi**2, device=device) - 2 * torch.arctan(torch.tensor(-0.25, device=device))**2)
    assert torch.isclose(theta(input4), torch.tensor(0.5 * torch.pi**2, device=device))

