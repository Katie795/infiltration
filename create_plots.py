import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os
import tomllib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    action="store",
    help="name of the parameter file",
    default="config.toml"
)

parser.add_argument(
    "--animate",
    action="store_true",
    help="whether to produce an animation of the water depth"
)

parser.add_argument(
    "--q",
    action="store_true",
    help="whether to plot the values of q"
)

parser.add_argument(
    "--domain",
    action="store_true",
    help="whether to produce plots of the domain"
)

parser.add_argument(
    "--animate_comparison",
    action="store_true",
    help="whether to produce an animation of the water depth comparing infiltration and no infiltration"
)

args, _ = parser.parse_known_args()

with open(args.config, "rb") as f:
    config = tomllib.load(f)

if args.animate:
    X = np.genfromtxt("data/x_coords.csv", delimiter=',')
    Y = np.genfromtxt("data/y_coords.csv", delimiter=',')

    deepest_arrray = np.genfromtxt(f"results/{config["files"]["eta_file"]}_{config["parameters"]["nt"]}.csv", delimiter=',')
    deepest = np.max(deepest_arrray)

    nfig = int(config["parameters"]["nt"]/ config["parameters"]["n_out"])

    for i in range(1, nfig):
        Z = np.genfromtxt(f"results/{config["files"]["eta_file"]}_{config["parameters"]["n_out"]*i}.csv", delimiter=',')
        Z_masked = np.ma.masked_where(Z < 0.00000001, Z)

        fig, ax = plt.subplots(1,1, figsize=(20,20))
        height = ax.pcolormesh(X, Y, Z_masked, shading="nearest", vmin=0, vmax=deepest)
        cb = plt.colorbar(height, label="water depth (m)")
        plt.savefig(f"plots/timestep_{i}.png")
        plt.close()


    images = []
    for i in range(1, nfig):
        images.append(imageio.imread(f"plots/timestep_{i}.png"))
        imageio.mimsave("plots/animation.gif", images)

    for i in range(1, nfig):
        os.remove(f"plots/timestep_{i}.png")

if args.animate_comparison:
    X = np.genfromtxt("data/x_coords.csv", delimiter=',')
    Y = np.genfromtxt("data/y_coords.csv", delimiter=',')

    deepest_arrray = np.genfromtxt(f"results/eta_impervious_{config["parameters"]["nt"]}.csv", delimiter=',')
    deepest = np.max(deepest_arrray)

    nfig = int(config["parameters"]["nt"]/ config["parameters"]["n_out"])

    for i in range(1, nfig):
        Z1 = np.genfromtxt(f"results/eta_impervious_{config["parameters"]["n_out"]*i}.csv", delimiter=',')
        Z2 = np.genfromtxt(f"results/eta_infiltration_{config["parameters"]["n_out"]*i}.csv", delimiter=',')
        Z_masked1 = np.ma.masked_where(np.isclose(Z1, 0.), Z1)
        Z_masked2 = np.ma.masked_where(np.isclose(Z1, 0.), Z2)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(55,25))
        height1 = ax1.pcolormesh(X, Y, Z_masked1, shading="nearest", vmin=0, vmax=deepest)
        height2 = ax2.pcolormesh(X, Y, Z_masked2, shading="nearest", vmin=0, vmax=deepest)
        ax1.set_title("No infiltration")
        ax2.set_title("Infiltration")
        cb = plt.colorbar(height2, label="water depth (m)", ax=[ax1, ax2])
        plt.savefig(f"plots/timestep_{i}.png")
        plt.close()

    images = []
    for i in range(1, nfig):
        images.append(imageio.imread(f"plots/timestep_{i}.png"))
        imageio.mimsave("plots/animation.gif", images)

    for i in range(1, nfig):
        os.remove(f"plots/timestep_{i}.png")




if args.domain:
    X = np.genfromtxt("data/rain_garden_indices.csv", delimiter=',')[1:,0 ]
    Y = np.genfromtxt("data/rain_garden_indices.csv", delimiter=',')[1:,1 ]
    fig, ax = plt.subplots(1,1, figsize=(20,20))
    plt.scatter(X,Y)
    plt.savefig(f"plots/testing_indices.png")
    plt.close()

    X = np.genfromtxt("data/rain_garden.csv", delimiter=',')[1:,0 ]
    Y = np.genfromtxt("data/rain_garden.csv", delimiter=',')[1:,1 ]
    fig, ax = plt.subplots(1,1, figsize=(20,20))
    plt.scatter(X,Y)
    plt.savefig(f"plots/testing_coords.png")
    plt.close()

    Z = np.genfromtxt(f"data/rain_garden_included.csv", delimiter=',')
    Z_masked = np.ma.masked_where(Z < -1000, Z)

    fig, ax = plt.subplots(1,1, figsize=(20,20))
    plt.xlim(0, 58.6) # set the maximum limits
    plt.ylim(0, 60.2) # set the maximum limits
    ax.pcolormesh(X, Y, Z_masked, shading="nearest")
    plt.savefig(f"plots/full_domain.png")
    plt.close()

if args.q:
    X = np.genfromtxt("data/x_coords.csv", delimiter=',')
    Y = np.genfromtxt("data/y_coords.csv", delimiter=',')
    q = np.genfromtxt("results/q.csv", delimiter=',')
    fig, ax = plt.subplots(1,1, figsize=(20,20))
    ax.pcolormesh(X, Y, q) 
    plt.savefig(f"plots/q.png")
    plt.close()
