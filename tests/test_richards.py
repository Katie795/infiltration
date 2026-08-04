# Tests for Richard's solver.
# 1: Tests whether padded coordinates line up with original coordinates
# 2: Tests whether interior dirichlet boundary function works for trivial case 
# 3: Tests whether Kappa produces correct output for various inputs
# 4: Tests whether theta produces correct output for various inputs
# 5: Tests whether uexact produces correct output for various inputs

import numpy as np
import torch
from AI4Flood.richards_solver import *

is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")

nx = 10
length = 10
t = 0

### generate the coordinates ###
generate_coords(nx, nx, name="horung_", length=length)
generate_padded_coords(nx, nx, name="horung_padded_", length=length)
x = torch.tensor(np.genfromtxt("../data/horung_x_coords.csv", delimiter=','))
y = torch.tensor(np.genfromtxt("../data/horung_y_coords.csv", delimiter=','))
x_padded = torch.tensor(np.genfromtxt("../data/horung_padded_x_coords.csv", delimiter=','))
y_padded = torch.tensor(np.genfromtxt("../data/horung_padded_y_coords.csv", delimiter=','))

### tests ###

def test_coordinates():
    """Test whether x_padded and y_padded align with x and y"""
    assert torch.equal(x, x_padded[1:-1, 1:-1])
    assert torch.equal(y, y_padded[1:-1, 1:-1])
    assert torch.equal(x[-1, -1], torch.tensor(length)) # test coords correspond to length

def test_boundaries():
    """Test whether uu aligns with u_ex, when uu is u_exact with applied boundary condition"""
    u_ex = u_exact(x_padded,y_padded,t)[0,0,1:nx+2,1:nx+2]
    uu = dirichlet_boundary_condition(u_exact(x,y,t), u_exact(x_padded,y_padded,t))

    assert torch.equal(u_ex, uu[0,0,1:nx+2,1:nx+2])

def test_kappa():
    """Test whether kappa produces correct output when input = -0.5, -0.25, 0.25, and 0.5. This covers a range
    of values for x - y when (x,y) are in the [0,1]x[0,1] domain"""
    input1 = torch.tensor([[[[-0.5]]]])
    input2 = torch.tensor([[[[0.5]]]])
    input3 = torch.tensor([[[[-0.25]]]])
    input4 = torch.tensor([[[[0.25]]]])

    assert Kappa(input1) == 1.6
    assert Kappa(input2) == 2
    assert Kappa(input3) == 1.88235294118
    assert Kappa(input4) == 2

def test_theta():
    """Test whether kappa produces correct output when input = -0.5, -0.25, 0.25, and 0.5. This covers a range
    of values for x - y when (x,y) are in the [0,1]x[0,1] domain"""
    input1 = torch.tensor([[[[-0.5]]]], device=device)
    input2 = torch.tensor([[[[0.5]]]], device=device)
    input3 = torch.tensor([[[[-0.25]]]])
    input4 = torch.tensor([[[[0.25]]]])

    assert torch.isclose(theta(input1), torch.tensor(0.5 * torch.pi**2, device=device) - 2 * torch.arctan(torch.tensor(-0.5, device=device))**2)
    assert torch.isclose(theta(input2), torch.tensor(0.5 * torch.pi**2, device=device))
    assert torch.isclose(theta(input3), torch.tensor(0.5 * torch.pi**2, device=device) - 2 * torch.arctan(torch.tensor(-0.25, device=device))**2)
    assert torch.isclose(theta(input4), torch.tensor(0.5 * torch.pi**2, device=device))

def test_uexact():
    """Test whether u_exact produces correct output when input = -0.5, -0.25, 0.25, and 0.5. This covers a range
    of values for x - y when (x,y) are in the [0,1]x[0,1] domain"""
    input1 = torch.tensor([[[[-0.5]]]])
    input2 = torch.tensor([[[[0.5]]]])
    input3 = torch.tensor([[[[-0.25]]]])
    input4 = torch.tensor([[[[0.25]]]])
    t = 0

    a = torch.tensor([[0]])
    b = torch.tensor([[0.25]])
    c = torch.tensor([[0.5]])
    

    assert u_exact(a, c, t) == 0.25
    assert torch.isclose(u_exact(c, a, t), torch.tensor(-0.249936250187829992230))
    assert u_exact(a, b, t) == 0.125
    assert torch.isclose(u_exact(b, a, t), torch.tensor(-0.1249979760882269638124333225981))
