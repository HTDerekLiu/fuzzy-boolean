import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator

def square_grid_bilinear_interpolator(data, val_min=-1, val_max=1):
    """
    create a bilinear interpolator for the data

    Input:
        data: (n*n) size np array
        val_min: (1,) min value of the grid point
        val_max: (1,) max value of the grid point

    Outputs
        interpolator: a object such that you can get interpolated data via "interpolator(pts)", pts are (m,2) query points 
    """
    grid_size = np.round(np.sqrt(len(data.flatten()))).astype(int)
    if grid_size**2 != len(data.flatten()):
        raise ValueError("In grid_bilinear_interpolator, grid size does not match data size")
    x = np.linspace(val_min, val_max, grid_size)
    bilinear_interpolator = RegularGridInterpolator((x, x), data.reshape(grid_size, grid_size))
    return bilinear_interpolator
