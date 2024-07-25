import torch

def sample_2D_grid(resolution, low = -1, high = 1):
    """
    sample unit image grid points between low ~ high

    Inputs
        resolution: a number of the resolution at each axis

    Outputs
        V: nx2 grid point locations
    """
    idx = torch.linspace(low,high,steps=resolution)
    x, y = torch.meshgrid(idx, idx, indexing='ij')
    V = torch.cat((x.reshape(-1).unsqueeze(1), y.reshape(-1).unsqueeze(1)), 1) 
    return V
