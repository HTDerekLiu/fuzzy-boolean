import torch

def compute_image_sdf(points, bilinear_interpolator):
    sdf = bilinear_interpolator(points)
    sdf_torch = torch.from_numpy(sdf).type(torch.float)
    return sdf_torch