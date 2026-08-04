import pandas as pd
import matplotlib.pyplot as plt
import torch

kappa_s = 0.05#0.0950     # saturated conductivity
theta_r = 0.3783     # minimum saturation percentage
theta_s = 0.6602     # maximum saturation percentage

df_kappa = pd.read_csv("../data_org/kappa(pF)2.csv") # import the data
df_theta = pd.read_csv("../data_org/theta(pF)2.csv")
pF_kappa = torch.tensor(df_kappa.to_numpy()[:, 0]) # convet to tensor
pF_theta = torch.tensor(df_theta.to_numpy()[:, 0])
kappa    = torch.tensor(df_kappa.to_numpy()[:, 1])
theta    = 0.01 * torch.tensor(df_theta.to_numpy()[:, 1]) # converting percentage to fraction

pressure_head_kappa = - 10 ** pF_kappa # potential force is logarithm of pressure head (cm)
u_kappa = 0.01 * pressure_head_kappa   # converting to m

pressure_head_theta = - 10 ** pF_theta # potential force is logarithm of pressure head (cm)
u_theta = 0.01 * pressure_head_theta   # converting to m

kappa = 10 ** kappa # data is in log(K).

theta_normalised = (theta - theta_r) / (theta_s - theta_r)

def mse(x,y):
    return torch.mean(torch.sum((x - y)**2))

def my_mse(params):
    f = find_theta_coefficients(*params)
    return mse(f, theta_normalised)

def find_theta_coefficients(alpha, n):
    m = (n - 1) / n
    theta1 = (1 / (1 + (- alpha * u_theta) ** n)) ** m
    return theta1

coeffs = torch.tensor([1.5162, 1.8313])
#coeffs.requires_grad_()

#for i in range(50):
#    loss = my_mse(coeffs)
#    loss.backward()
#    with torch.no_grad():
#        coeffs -= coeffs.grad * 0.1
#    print(f"Step {i}, loss = {loss:.2f}, coeffs = {coeffs}")

def theta_function(u):
    alpha = coeffs[0]
    n = coeffs[1]
    m = (n - 1) / n
    theta1 = (1 / (1 + (- alpha * u) ** n)) ** m
    ans = torch.where(u < 0, theta1, 1)
    return ans

def kappa_function(u):
    n = coeffs[1]
    m = (n - 1) / n
    kappa1 = theta_function(u) ** 0.5 * (1 - (1 - theta_function(u) ** (1 / m)) ** (1 / m)) ** 2
    ans = kappa_s * torch.where(u < 0, kappa1, 1)
    return ans

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.scatter(u_theta, theta_normalised)
ax.plot(u_theta, theta_function(u_theta).detach().numpy(), color="red")
ax.set_xlabel(r"Capillary potential $u$")
ax.set_ylabel(r"$\theta(u)$")
plt.tight_layout()
plt.savefig("../plots/theta_u.png")
plt.close()

fig, ax = plt.subplots(1,1, figsize=(5,5))
ax.scatter(u_kappa, kappa )
ax.plot(u_kappa, kappa_function(u_kappa), color="red")
ax.set_xlabel(r"Capillary potential $u$")
ax.set_ylabel(r"$\kappa(u)$")
plt.tight_layout()
plt.savefig("../plots/kappa_u.png")

