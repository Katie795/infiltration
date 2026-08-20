# Tests for the horung messing test coefficients

import torch
from AI4Flood.richards_solver import *
from AI4Flood.generate_csv_files import *
from AI4Flood.richards_coefficients import u_exact_hm, kappa_hm, theta_hm
from AI4Flood.operators import get_weights_linear_2D
torch.set_default_dtype(torch.float64)

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
    input1 = torch.tensor([[[[-0.5]]]])
    input2 = torch.tensor([[[[0.5]]]])
    input3 = torch.tensor([[[[-0.25]]]])
    input4 = torch.tensor([[[[0.25]]]])

    assert torch.isclose(theta_hm(input1), torch.tensor(0.5 * torch.pi**2,) - 2 * torch.arctan(torch.tensor(-0.5))**2)
    assert torch.isclose(theta_hm(input2), torch.tensor(0.5 * torch.pi**2))
    assert torch.isclose(theta_hm(input3), torch.tensor(0.5 * torch.pi**2)) - 2 * torch.arctan(torch.tensor(-0.25)**2)
    assert torch.isclose(theta_hm(input4), torch.tensor(0.5 * torch.pi**2))

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

def test_diffusion_operator():
    length= 10
    nx = 100
    dx = length / nx

    x, y, xx, yy = generate_csv_files(nx, nx, length, length, name="horung_messsing")
    t = 0

    [w_diff, wm1, diag, bias] = get_weights_linear_2D(dx)
            
    diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
    cmm  = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
    diff.weight.data = w_diff
    cmm.weight.data = wm1
    diff.bias.data = bias
    cmm.bias.data = bias

    u = u_exact_hm(x, y, t)
    uu = u_exact_hm(xx, yy, t)

    i, j =25, 25

    print("x      =", x[i,j].item())
    print("y      =", y[i,j].item())
    print("s      =", (x-y)[i,j].item())
    print("u      =", u[0,0, i,j].item())
    print("kappa  =", kappa_hm(u)[0,0,i,j].item())

    analytic = -2 * u / (1 + u**2)**2

    term1 = diff(kappa_hm(uu) * uu)
    term2 = kappa_hm(u) * diff(uu)
    term3 = u * diff(kappa_hm(uu))

    print("analytic =", analytic[0,0,i,j].item())
    print("term 1   =", term1[0,0,i,j].item())
    print("term 2   =", term2[0,0,i,j].item())
    print("term 3   =", term3[0,0,i,j].item())
    print(
        "numerical =",
        (0.5 * (term1 + term2 - term3))[0,0,i,j].item()
    )
    div_k_grad_ui = 0.5 * (diff(kappa_hm(uu) * uu)  \
                            + kappa_hm(u) * diff(uu) \
                            - u * diff(kappa_hm(uu))) # updating diffusion term

    #print(analytic[0, 0, 25, 75])
    #print(div_k_grad_ui[0, 0, 25, 75])

test_diffusion_operator()