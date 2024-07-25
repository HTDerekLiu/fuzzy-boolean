import sys, os
sys.path.append("../")
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np

from model import csg_tree_full_binary, csg_tree_pointers
from utils.binary_image_to_sdf_interpolator import binary_image_to_sdf_interpolator
from utils.sample_image_points import sample_image_points
from utils.compute_image_occupancy import compute_image_occupancy
from utils.plot_2D_occupancy import plot_2D_occupancy
from utils.sample_2D_grid import sample_2D_grid

torch.manual_seed(0)
device = "cpu"

# parameters
resolution = 64 # for ploting
num_samples = 1024 # for optimiztion
depth = 7 # csg tree depth

# load image
bilinear_interpolator = binary_image_to_sdf_interpolator("./github_logo.jpg", 256)

# create output folder
folder = "./output/"
if not os.path.exists(folder):
   os.makedirs(folder)

# save the ground truth image
V = sample_2D_grid(resolution)
plt.figure(0)
plt.clf()
plot_2D_occupancy(compute_image_occupancy(V, bilinear_interpolator))
plt.colorbar()
plt.savefig(folder + "gt.png")

# initialize the model
model = csg_tree_full_binary(depth).to(device)

# loss function
loss_func = torch.nn.MSELoss()
params = [
    {'params': model.parameters()},
]
optimizer = torch.optim.Adam(params, lr=1e-3)

# optimization variables
resample_every_epoch = 10
save_every_epoch = 1000
num_epochs = 100000
pbar = tqdm(range(num_epochs))
total_loss_history = []

# move toe device
V = V.to(device)
for epoch in pbar:

    # sample ground truth points
    if (epoch) % resample_every_epoch == 0:
        P = sample_image_points(num_samples, bilinear_interpolator)
        gt = compute_image_occupancy(P, bilinear_interpolator)
        P = P.unsqueeze(0).to(device)
        gt = gt.to(device)

    # forward
    out = model(P)
    loss = loss_func(out.squeeze(0), gt)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # keep track losses
    total_loss_val = loss.item()
    total_loss_history.append(total_loss_val)   
    pbar.set_postfix({'total': total_loss_val})

    # save intermediate results
    if (epoch+1) % save_every_epoch == 0:
        out = model(V.unsqueeze(0))
        out = out.squeeze(0)

        plt.figure(2)
        plt.clf()
        plot_2D_occupancy(out)
        plt.colorbar()
        plt.savefig(folder + "reconstruction.png")

        plt.figure(3)
        plt.clf()
        plt.semilogy(total_loss_history)
        plt.savefig(folder + "loss_history.png")

        torch.save(model, folder + "model_params.pt")

# convert the output to be a true CSG tree with pruning
model = torch.load(folder + "model_params.pt", map_location=torch.device('cpu'))
csg_tree = csg_tree_pointers(model)

csg_tree.pruning(V, threshold_similarity = 1e-3) 
csg_tree.print_tree(folder + "tree.txt")
out = csg_tree.forward(V)
csg_tree.save_all_primitives(folder + "prims_after_prune.png", V)

plt.figure()
plt.clf()
plot_2D_occupancy(out)
plt.colorbar()
plt.savefig(folder + "reconstruction_after_prune.png")
