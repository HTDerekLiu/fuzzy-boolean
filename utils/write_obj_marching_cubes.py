import mcubes
import numpy as np
from .to_numpy import to_numpy
import torch

def write_obj_marching_cubes(fileName, u, isoval = 0.0, run_smoothing=False):
    """
    using marching cubes to extract the mesh and save it. the corresponding points of "u" should be generated using sample_unit_grid.py or having the same order

    Example usage:
        resolution = 16
        x = sample_unit_grid(resolution)
        out = sdf(x)
        out = out.cpu().detach().numpy()
        write_obj_marching_cubes("output.obj", out, 0.0)
    """
    if torch.sign(u.max()) == torch.sign(u.min()):
        raise ValueError("input implicit function does not have zero-isoline")
    u = u.flatten()
    dim = np.round(u[:].shape[0] ** (1./3)).astype(int)
    u = u.reshape(dim, dim, dim)
    if torch.is_tensor(u):
        u = to_numpy(u)
    else:
        raise ValueError("input to write_obj_marching_cubes is not a torch tensor")
    if run_smoothing:
        u = mcubes.smooth(u)
    vertices, triangles = mcubes.marching_cubes(u, isoval)
    triangles[:,[0,1]] = triangles[:,[1,0]]
    mcubes.export_obj(vertices, triangles, fileName)