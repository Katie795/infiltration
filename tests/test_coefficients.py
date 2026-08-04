import numpy as np
import torch
from AI4Flood.richards_solver import *
from AI4Flood.generate_csv_files import *
from AI4Flood.richards_coefficients import u_exact_hm, kappa_hm, theta_hm

def test_kappa():
    """Test whether kappa produces correct output when input = -0.5, -0.25, 0.25, and 0.5. This covers a range
    of values for x - y when (x,y) are in the [0,1]x[0,1] domain"""
    input1 = torch.tensor([[[[-0.5]]]])
    input2 = torch.tensor([[[[0.5]]]])
    input3 = torch.tensor([[[[-0.25]]]])
    input4 = torch.tensor([[[[0.25]]]])

    assert kappa_hm(input1) == 1.6
    assert kappa_hm(input2) == 2
    assert kappa_hm(input3) == 1.88235294118
    assert kappa_hm(input4) == 2

def test_theta():
    """Test whether kappa produces correct output when input = -0.5, -0.25, 0.25, and 0.5. This covers a range
    of values for x - y when (x,y) are in the [0,1]x[0,1] domain"""
    input1 = torch.tensor([[[[-0.5]]]], device=device)
    input2 = torch.tensor([[[[0.5]]]], device=device)
    input3 = torch.tensor([[[[-0.25]]]])
    input4 = torch.tensor([[[[0.25]]]])

    assert torch.isclose(theta_hm(input1), torch.tensor(0.5 * torch.pi**2, device=device) - 2 * torch.arctan(torch.tensor(-0.5, device=device))**2)
    assert torch.isclose(theta_hm(input2), torch.tensor(0.5 * torch.pi**2, device=device))
    assert torch.isclose(theta_hm(input3), torch.tensor(0.5 * torch.pi**2, device=device) - 2 * torch.arctan(torch.tensor(-0.25, device=device))**2)
    assert torch.isclose(theta_hm(input4), torch.tensor(0.5 * torch.pi**2, device=device))

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
    

    assert u_exact_hm(a, c, t) == 0.25
    assert torch.isclose(u_exact_hm(c, a, t), torch.tensor(-0.249936250187829992230))
    assert u_exact_hm(a, b, t) == 0.125
    assert torch.isclose(u_exact_hm(b, a, t), torch.tensor(-0.1249979760882269638124333225981))
