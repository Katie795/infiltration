import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os



X = np.genfromtxt("data/x_coords.csv", delimiter=',')
Y = np.genfromtxt("data/y_coords.csv", delimiter=',')

Z = np.genfromtxt(f"data_generated/rain_garden_included.csv", delimiter=',')
Z_masked = np.ma.masked_where(Z < -1000, Z)

fig, ax = plt.subplots(1,1, figsize=(20,20))
plt.xlim(0, 58.6) # set the maximum limits
plt.ylim(0, 60.2) # set the maximum limits
ax.pcolormesh(X, Y, Z_masked, shading="nearest")
plt.savefig(f"plots/full_domain.png")
plt.close()



'''

for i in range(1,100):
    Z = np.genfromtxt(f"results/eta_road_{10*i}.csv", delimiter=',')
    Z_masked = np.ma.masked_where(Z < 0.00000001, Z)

    fig, ax = plt.subplots(1,1, figsize=(20,20))
    plt.xlim(0, 58.6) # set the maximum limits
    plt.ylim(0, 60.2) # set the maximum limits
    ax.pcolormesh(X, Y, Z_masked, shading="nearest")
    plt.savefig(f"plots/timestep_{i}.png")
    plt.close()


images = []
for i in range(1,20):
    images.append(imageio.imread(f"plots/timestep_{i}.png"))
    imageio.mimsave("plots/animation.gif", images)

for i in range(1,100):
    os.remove(f"plots/timestep_{i}.png")
'''

X = np.genfromtxt("data_generated/rain_garden_indices.csv", delimiter=',')[1:,0 ]
Y = np.genfromtxt("data_generated/rain_garden_indices.csv", delimiter=',')[1:,1 ]
fig, ax = plt.subplots(1,1, figsize=(20,20))
plt.scatter(X,Y)
plt.savefig(f"plots/testing_indices.png")
plt.close()

X = np.genfromtxt("data_generated/rain_garden.csv", delimiter=',')[1:,0 ]
Y = np.genfromtxt("data_generated/rain_garden.csv", delimiter=',')[1:,1 ]
fig, ax = plt.subplots(1,1, figsize=(20,20))
plt.scatter(X,Y)
plt.savefig(f"plots/testing_coords.png")
plt.close()