import torch

def normalize_to_unit_box(V, margin = 0.0):
    """
    this function normalize a shape to a bounding box between -1 and 1. One can specify margin. For example, if margin = 0.3, then V will be bounded by -0.7 - 0.7.

    Inputs:
        V: n x 3 array of vertex locations
        margin: a scalar between 0, 1
    Outputs
        V: n x 3 array of normalized vertex locations
    """
    if margin > 1.0 or margin < 0:
        raise ValueError("margin in normalize_to_unit_box must be 0~1")

    # move the mesh to the center
    bbx_mean = (torch.max(V,0)[0] + torch.min(V,0)[0]) / 2.0
    V = V - bbx_mean[None,:]

    # normalize it to bounding box with margin
    V = V / torch.abs(V).max()
    V = V * (1.0 - margin)
    return V