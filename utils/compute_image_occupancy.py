import torch

def compute_image_occupancy(points, bilinear_interpolator):
    sdf = bilinear_interpolator(points)
    sdf_torch = torch.from_numpy(sdf).type(torch.float)
    return torch.sign(-sdf_torch) / 2. + 0.5