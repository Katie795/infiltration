### Coefficients for Richards equation ###
import torch

def theta_hm(s):
    """theta(s) in the Horung-Messing test. Here
    theta(s) = pi**2/2 - 2 arctan(s)**2 if s < 0,
             = pi**2/2                  if s > 0.
    
    :args s: input where s = x - y - t
    """
    s = s[0, 0]  # still assumes (1,1,n,n)

    base = 0.5 * torch.pi**2
    atan_term = 2 * torch.atan(s)**2

    ans = torch.where(s < 0, base - atan_term, base)

    return ans.unsqueeze(0).unsqueeze(0).to(torch.float64)


def theta_prime_hm(s):
    """theta'(s) in the Horung-Messing test. Needed for Newton's method. Here
    theta'(s) = -4 arctan(s) / (s**2 + 1) if s < 0,
              = 0                         if s > 0.
    
    :args s: input where s = x - y - t
    """
    s = s[0, 0]  # still assumes (1,1,n,n)

    base = 0
    theta_left = - 4 * torch.atan(s)/ (s**2 + 1)
    ans = torch.where(s < 0, theta_left, base)

    return torch.unsqueeze(torch.unsqueeze(ans, 0),0).to(torch.float64)


def kappa_hm(s):
    """kappa_hm(s) in the Horung-Messing test. Here
    kappa_hm(s) = 2 / (1 + s**2) if s < 0,
             = 2              if s > 0.
    
    :args s: input where s = x - y - t
    """
    s = s[0, 0]

    base = 2.0
    kappa_left = 2.0 / (1.0 + s ** 2) 

    ans = torch.where(s < 0, kappa_left, base)

    return ans.unsqueeze(0).unsqueeze(0).to(torch.float64)

def Kappa_prime(s):
    """kappa_hm'(s) in the Horung-Messing test. Needed for Newton's method. Here
    kappa_hm'(s) = -4s / (1 + s**2)**2 if s < 0,
              = 0                   if s > 0.
    
    :args s: input where s = x - y - t
    """
    s = s[0, 0]

    base = 0
    kappa_prime_left = -4 * s / (1 + s**2)**2

    ans = torch.where(s < 0, kappa_prime_left, base)

    return ans.unsqueeze(0).unsqueeze(0).to(torch.float64)

def u_exact_hm(x, y, t):
    """u_exact in the Horung-Messing test.  Here
    u_exact(x,y,t) = -1/2 s                     if s < 0,
                   = - tan((e^s - 1)/(e^s + 1)) if s > 0.
    
    :args x: x coordinates
    :args y: y coordinates
    :args t: time
    """
    s = x - y - t

    left = -0.5 * s
    right = -torch.tan(torch.tanh(s / 2))

    ans = torch.where(s < 0, left, right)

    return ans.unsqueeze(0).unsqueeze(0).to(torch.float64)


def theta_malago(u):
    alpha = 1.5162
    n = 1.8313
    m = (n - 1) / n
    theta1 = (1 / (1 + (- alpha * u) ** n)) ** m
    ans = torch.where(u < 0, theta1, 1)
    return ans.to(torch.float64)

def kappa_malago(u):
    n = 1.8313
    kappa_s = 0.05
    m = (n - 1) / n
    kappa1 = theta_malago(u) ** 0.5 * (1 - (1 - theta_malago(u) ** (1 / m)) ** (1 / m)) ** 2
    ans = kappa_s * torch.where(u < 0, kappa1, 1)
    return ans.to(torch.float64)

if __name__ == "__main__":
    import generate_csv_files
    import numpy as np
    from misc_functions import plot_tensor
    """Here we are plotting the output of these functions"""
    length = 1
    nx = 10
    t = 0

    input_shape_pd = (1, 1, nx + 3, nx + 3)
    uu = torch.zeros(input_shape_pd)

    # Checking that the coordinates are correct - do we really need an extra 1?
    generate_csv_files.generate_coords(nx, nx, name="horung_", length=length)
    generate_csv_files.generate_padded_coords(nx, nx, name="horung_padded_", length=length)
    x = torch.tensor(np.genfromtxt("data/horung_x_coords.csv", delimiter=','))
    y = torch.tensor(np.genfromtxt("data/horung_y_coords.csv", delimiter=','))
    x_padded = torch.tensor(np.genfromtxt("data/horung_padded_x_coords.csv", delimiter=','))
    y_padded = torch.tensor(np.genfromtxt("data/horung_padded_y_coords.csv", delimiter=','))

    s = torch.unsqueeze(torch.unsqueeze(x - y - t, 0), 0) 



    plot_tensor(x, y, u_exact(x,y,t)[0,0,:,:].cpu(), "u_exact")
    plot_tensor(x, y, Kappa(s)[0,0,:,:].cpu(), "Kappa")
    plot_tensor(x, y, theta(s)[0,0,:,:].cpu(), "theta")

    uu = dirichlet_boundary_condition(u_exact(x,y,t), u_exact(x_padded,y_padded,t))
    plot_tensor(x_padded, y_padded, uu[0,0,:,:].cpu(), "dirichlet_bcs")
    plot_tensor(x, y, u_exact(x_padded,y_padded,t)[0,0,1:nx+2,1:nx+2].cpu() - uu[0,0,1:nx+2,1:nx+2].cpu(), "dirichlet_bcs_error")
