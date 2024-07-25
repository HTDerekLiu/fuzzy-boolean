import torch

def sample_3D_grid(resolution, min=-1, max=1):
    """
    sample unit voxel grid points between -0.5, 0.5

    Inputs
        resolution: a number of the resolution at each axis

    Outputs
        V: nx3 grid point locations
    """
    idx = torch.linspace(min,max,steps=resolution)
    x, y, z = torch.meshgrid(idx, idx, idx, indexing='ij')
    V = torch.cat((x.reshape(-1).unsqueeze(1), y.reshape(-1).unsqueeze(1), z.reshape(-1).unsqueeze(1)), 1) 
    return V
