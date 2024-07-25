import torch
from .compute_image_sdf import compute_image_sdf

def sample_image_points(num_samples, bilinear_interpolator, softness = 5):
    def random_samples(num_samples):
        return (torch.rand((num_samples,2)) -0.5) * 2.0
    
    P_uniform = random_samples(num_samples // 4)

    # sample the first time
    P = random_samples(num_samples)
    prob = torch.abs(compute_image_sdf(P, bilinear_interpolator)) / softness
    prob = torch.exp(1./(prob+2e-1))
    prob = prob / prob.max()
    idx = torch.where(torch.bernoulli(prob))[0]
    P = torch.vstack((P_uniform, P[idx,:]))

    while P.shape[0] < num_samples:
        P_new = random_samples(num_samples)
        prob = torch.abs(compute_image_sdf(P_new, bilinear_interpolator)) / softness
        prob = torch.exp(1./(prob+2e-1))
        prob = prob / prob.max()
        idx = torch.where(torch.bernoulli(prob))[0]
        P = torch.vstack((P, P_new[idx]))
    return P[:num_samples,:]
