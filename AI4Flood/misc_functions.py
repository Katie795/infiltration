import os, shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def clear_output_folder():
    folder = '/home/km2560/Workspace/Flooding/results'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    return

def plot_tensor(x, y, tensor, name):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    height = ax.pcolormesh(x, y, tensor)
    cb = plt.colorbar(height, label=name, ax=ax)
    plt.title(name)
    plt.savefig(f"../plots/{name}")

def plot_tensor_infiltration(x, y, tensor, name):
    fig, ax = plt.subplots(1,1, figsize=(16,4.8))
    levels = MaxNLocator(nbins=15).tick_values(tensor.min(), tensor.max())
    cmap = plt.colormaps['PiYG']
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    height = ax.pcolormesh(x, y, tensor, cmap=cmap)
    cb = plt.colorbar(height, label=name, ax=ax)
    plt.title(name)
    plt.savefig(f"../plots/{name}")