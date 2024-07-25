import torch

def to_numpy(x):
    return x.detach().cpu().numpy()
    