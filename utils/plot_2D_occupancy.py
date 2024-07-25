import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from .to_numpy import to_numpy

def plot_2D_occupancy(u):
    u = to_numpy(u.flatten())
    grid_resolution = np.round(u[:].shape[0] ** (1./2)).astype(int)
    levels = np.linspace(-1e-5, 1+1e-5, 101)

    colormap = matplotlib.colors.LinearSegmentedColormap.from_list('SDF', [(0.0,'#92c5de'),(0.5, '#ffffff'), (1.0,'#ef8a62')], N=256) # red blue

    plot = plt.contourf(u.reshape(grid_resolution,grid_resolution).T, levels = levels, cmap=colormap, origin="lower")
    for c in plot.collections:
        c.set_edgecolor("face")
    plt.xticks([])
    plt.yticks([])
    plt.axis('equal')
    plt.axis('off')