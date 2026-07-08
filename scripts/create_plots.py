import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os
import tomllib
import argparse
import csv
import datetime

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
    "--filename",
    action="store",
    help="whether to produce an animation of the water depth",
    default="eta_infiltration"
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
    "--rainfall_data",
    action="store_true",
    help="whether to produce plots of the rainfall data"
)

parser.add_argument(
    "--animate_malago",
    action="store_true",
    help="whether to produce aniamte malago infiltration"
)

parser.add_argument(
    "--animate_comparison",
    action="store_true",
    help="whether to produce an animation of the water depth comparing infiltration and no infiltration"
)

parser.add_argument(
    "--nfig",
    action="store",
    help="how many pictures to include in animation"
)

args, _ = parser.parse_known_args()

with open(args.config, "rb") as f:
    config = tomllib.load(f)

if args.animate:
    X = np.genfromtxt("../data/x_coords.csv", delimiter=',')
    Y = np.genfromtxt("../data/y_coords.csv", delimiter=',')

    deepest_arrray = np.genfromtxt(f"../results/{args.filename}_{config["parameters"]["nt"]}.csv", delimiter=',')
    deepest = np.max(deepest_arrray)

    nfig = int(config["parameters"]["nt"]/ config["parameters"]["n_out"])

    for i in range(1, nfig):
        Z = np.genfromtxt(f"../results/{args.filename}_{config["parameters"]["n_out"]*i}.csv", delimiter=',')
        Z_masked = np.ma.masked_where(Z < 0.00000001, Z)

        fig, ax = plt.subplots(1,1, figsize=(20,20))
        height = ax.pcolormesh(X, Y, Z_masked, shading="nearest", vmin=0, vmax=deepest)
        cb = plt.colorbar(height, label="water depth (m)")
        plt.savefig(f"../plots/timestep_{i}.png")
        plt.close()


    images = []
    for i in range(1, nfig):
        images.append(imageio.imread(f"../plots/timestep_{i}.png"))
        imageio.mimsave("../plots/animation.mp4", images)

    for i in range(1, nfig):
        os.remove(f"../plots/timestep_{i}.png")

if args.animate_malago:

    deepest_arrray = np.genfromtxt(f"../results/u_{args.nfig}.csv", delimiter=',')
    deepest = np.max(deepest_arrray)

    X = np.genfromtxt("../data/infiltration_x_coords.csv", delimiter=',')
    Y = np.genfromtxt("../data/infiltration_y_coords.csv", delimiter=',')

    for i in range(1, int(args.nfig)):
        Z = np.genfromtxt(f"../results/u_{i}.csv", delimiter=',')
        fig, ax = plt.subplots(1,1, figsize=(16,4.8))
        cmap = plt.colormaps['PiYG']

        height = ax.pcolormesh(X, Y, Z, shading="nearest", vmin=0, vmax=deepest, cmap=cmap)
        cb = plt.colorbar(height, label="pressure head")
        plt.title(f"Time = {10 * i} s")
        plt.savefig(f"../plots/u_{i}.png")
        plt.close()


    images = []
    for i in range(1, int(args.nfig)):
        images.append(imageio.imread(f"../plots/u_{i}.png"))
        imageio.mimsave("../plots/malago_infiltration.mp4", images)

    for i in range(1, int(args.nfig)):
        os.remove(f"../plots/u_{i}.png")

if args.animate_comparison:
    X = np.genfromtxt("../data/x_coords.csv", delimiter=',')
    Y = np.genfromtxt("../data/y_coords.csv", delimiter=',')

    deepest_arrray = np.genfromtxt(f"../results/u_{config["parameters"]["nt"]}.csv", delimiter=',')
    deepest = np.max(deepest_arrray)

    nfig = int(config["parameters"]["nt"]/ config["parameters"]["n_out"])

    for i in range(1, nfig):
        Z1 = np.genfromtxt(f"../results/u_{config["parameters"]["n_out"]*i}.csv", delimiter=',')
        Z2 = np.genfromtxt(f"../results/v_{config["parameters"]["n_out"]*i}.csv", delimiter=',')
        Z_masked1 = np.ma.masked_where(np.isclose(Z1, 0.), Z1)
        Z_masked2 = np.ma.masked_where(np.isclose(Z1, 0.), Z2)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(55,25))
        height1 = ax1.pcolormesh(X, Y, Z_masked1, shading="nearest", vmin=0, vmax=deepest, cmap="magma")
        height2 = ax2.pcolormesh(X, Y, Z_masked2, shading="nearest", vmin=0, vmax=deepest, cmap="magma")
        ax1.set_title(r"Speed in $x$ direction")
        ax2.set_title(r"Speed in $y$ direction")
        cb = plt.colorbar(height2, label="water depth (m)", ax=[ax1, ax2])
        plt.savefig(f"../plots/timestep_{i}.png")
        plt.close()

    images = []
    for i in range(1, nfig):
        images.append(imageio.imread(f"../plots/timestep_{i}.png"))
        imageio.mimsave("../plots/comparison_animation_uv.mp4", images)

    for i in range(1, nfig):
        os.remove(f"../plots/malago_{i}.png")


if args.rainfall_data:
    rainfall = np.genfromtxt("../data/Clifton_rain_modifed_for_rainfall_event.csv", delimiter=',')[1:, 0]
    time = np.genfromtxt("../data/Clifton_rain_modifed_for_rainfall_event.csv", delimiter=',')[1:, 1]
    with open("../data/Clifton_rain_modifed_for_rainfall_event.csv", newline="") as f:
        reader = csv.reader(f)
        data = list(reader)
        rainfall_time = [column[1] for column in data][1:]
    print(rainfall_time)

    newtime = np.array(range(len(rainfall)))
    fig, ax = plt.subplots(1,1)

    plt.plot(0.25 * newtime, rainfall)
    plt.title("Rainfall event on 5th January 2025")
    plt.xlabel("Hours")
    plt.ylabel("Rainfall (mm)")
    plt.savefig("../plots/rainfall_data.png")
    plt.show()



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
