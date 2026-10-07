import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os
import tomli as tomllib
import argparse
import csv
import datetime
import torch
import matplotlib.ticker as ticker

from AI4Flood.misc_functions import *
from AI4Flood.richards_coefficients import kappa_malago
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    action="store",
    help="name of the parameter file",
    default="config_rain_garden.toml"
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
    "--potential",
    action="store_true",
    help="whether to produce aniamte malago infiltration"
)

parser.add_argument(
    "--flux",
    action="store_true",
    help="whether to produce plots with water flux"
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

n_out = torch.tensor([config["IO"]["n_out"]])
dt    = torch.tensor([config["time"]["dt"]])
nx    = torch.tensor([config["mesh"]["nx"]]) 
ny    = torch.tensor([config["mesh"]["ny"]])
length   = torch.tensor([config["geometry"]["length"]]) 
depth    = torch.tensor([config["geometry"]["depth"]]) 
dx = length / nx
dy = depth / ny


if args.animate:
    X = np.genfromtxt("../data/x_coords.csv", delimiter=',')
    Y = np.genfromtxt("../data/y_coords.csv", delimiter=',')

    deepest_arrray = np.genfromtxt(f"../results/{args.filename}_{config['time']['nt']}.csv", delimiter=',')
    deepest = np.max(deepest_arrray)

    nfig = int(config["time"]["nt"]/ config["IO"]["n_out"])

    for i in range(1, nfig):
        Z = np.genfromtxt(f"../results/{args.filename}_{config['IO']['n_out']*i}.csv", delimiter=',')
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

if args.potential:

    # Obtain maximum and minimum values for capillary potential. Maximum will be in original thing
    potential1 = np.genfromtxt(f"../results/u_0.csv", delimiter=',')
    potential2 = np.genfromtxt(f"../results/u_{args.nfig}.csv", delimiter=',')

    min_p = np.min(potential1)
    max_p = np.max(potential2)
    levels = np.linspace(min_p, max_p, 5)
    plt.rcParams['contour.negative_linestyle'] = 'solid'


    X = np.genfromtxt("../data/infiltration_x_coords.csv", delimiter=',')
    Y = np.genfromtxt("../data/infiltration_y_coords.csv", delimiter=',')

    for i in range(0, int(args.nfig) + 1):
        Z = np.genfromtxt(f"../results/u_{i}.csv", delimiter=',')
        fig, ax = plt.subplots(1,1)
        cmap = plt.colormaps['inferno']

        #vmin=min_p, vmax=max_p, 

        height = ax.pcolormesh(X, Y, Z, shading="nearest", cmap=cmap)
        cb = plt.colorbar(height, label=r"Capillary potential $u$")
        cs = ax.contour(X, Y, Z, levels, linewidths=1, colors="black")
        ax.clabel(cs,  fontsize=9)
        #ax.set_title("Capillary potential in Malago road rain garden")
        ax.set_xlabel("Length (m)")
        ax.set_ylabel("Depth (m)")
        # time is saved every 100 timesteps where each timestep is 0.01 second

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        plt.text(0.80, 1.10, f"Time = {(dt * n_out * i).numpy()[0]} s", transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.tight_layout()

        plt.savefig(f"../plots/u_{i}.png")
        plt.close()

    images = []
    for i in range(0, int(args.nfig) + 1):
        images.append(imageio.imread(f"../plots/u_{i}.png"))
        imageio.mimsave("../plots/malago_infiltration_4.mp4", images)

    #for i in range(0, int(args.nfig) + 1):
    #    os.remove(f"../plots/u_{i}.png")

if args.flux:
    X = np.genfromtxt("../data/infiltration_x_coords.csv", delimiter=',')[1:, 1:]
    Y = np.genfromtxt("../data/infiltration_y_coords.csv", delimiter=',')[1:, 1:]
    Z = np.genfromtxt(f"../results/u_0.csv", delimiter=',')
    x = np.linspace(0, 4, nx)
    y = np.linspace(0, 1.2, ny)

    for i in range(0, int(args.nfig) + 1):
        Z = torch.tensor(np.genfromtxt(f"../results/u_{i}.csv", delimiter=','))
        grad_x, grad_y = calculate_grad(Z, dx, dy)
        fig, ax = plt.subplots(1,1)

        q_x = - kappa_malago(Z[:-1, :-1]) * grad_x       # water flux in the x direction
        q_y = - kappa_malago(Z[:-1, :-1]) * (grad_y + 1) # water flux in the y direction. PLus one for gravity
        q = ax.quiver(X[::3, ::3], Y[::3, ::3], q_x[::3, ::3], q_y[::3, ::3])
        #plt.title("Infiltration flux")
        plt.tight_layout()
        plt.savefig(f"../plots/u_quiver_{i}.png")
        plt.close()

        images = []
    #for i in range(0, int(args.nfig) + 1):
    #    images.append(imageio.imread(f"../plots/u_quiver_{i}.png"))
    #    imageio.mimsave("../plots/malago_quiver.mp4", images)

    #for i in range(0, int(args.nfig) + 1):
    #    os.remove(f"../plots/u_quiver_{i}.png")


if args.animate_comparison:
    X = np.genfromtxt("../data/x_coords.csv", delimiter=',')
    Y = np.genfromtxt("../data/y_coords.csv", delimiter=',')

    deepest_arrray = np.genfromtxt(f"../results/u_{config['time']['nt']}.csv", delimiter=',')
    deepest = np.max(deepest_arrray)

    nfig = int(config['time']['nt']/ config['IO']['n_out'])

    for i in range(1, nfig):
        Z1 = np.genfromtxt(f"../results/u_{config['IO']['n_out']*i}.csv", delimiter=',')
        Z2 = np.genfromtxt(f"../results/v_{config['IO']['n_out']*i}.csv", delimiter=',')
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
    rainfall = np.genfromtxt(config['IO']['rainfall_data'], delimiter=',')[1:, 0]
    time = np.genfromtxt(config['IO']['rainfall_data'], delimiter=',')[1:, 1]
    with open(config['IO']['rainfall_data'], newline="") as f:
        reader = csv.reader(f)
        data = list(reader)
        rainfall_time = [column[1][9:] for column in data][1:]
    print(rainfall_time)

    newtime = np.array(range(len(rainfall)))
    print(newtime)
    fig, ax = plt.subplots(1,1)
    ax.xaxis.set_major_locator(ticker.AutoLocator())

    plt.bar(rainfall_time, rainfall)
    plt.title("Rainfall event on 5th-6th January 2025")
    plt.xlabel("Time")
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
