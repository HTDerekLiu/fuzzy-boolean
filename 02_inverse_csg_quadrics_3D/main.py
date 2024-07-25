import sys, os
sys.path.append("../")
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse

from model import csg_net

from utils.read_obj import read_obj
from utils.normalize_to_unit_box import normalize_to_unit_box
from utils.single_mesh_dataloader import single_mesh_dataloader
from utils.sample_3D_grid import sample_3D_grid
from utils.write_obj_marching_cubes import write_obj_marching_cubes

def train():
    torch.manual_seed(0)

    DEVICE = torch.device("cpu")
    num_cpus = 1
    mesh_name = "table_51016.obj"
    depth = 10
    max_temperature = 100.
    boolean_frequency = 10.0

    folder = "./output/"
    if not os.path.exists(folder):
       os.makedirs(folder)

    U,F = read_obj(mesh_name)
    U = normalize_to_unit_box(U, 0.2)

    lr = 5e-4
    num_samples = 1024 * 100
    samples_per_batch = 1024
    num_epochs = 2000
    save_every_epochs = 500
    regenerate_training_data_every_epochs = 10

    dataloader = single_mesh_dataloader(U,F,num_samples = num_samples, samples_per_batch = samples_per_batch, num_workers = num_cpus)
    
    model = csg_net(depth, max_temperature, boolean_frequency).to(DEVICE)

    params = [
        {'params': model.parameters()},
    ]
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_func = torch.nn.MSELoss()

    resolution = 64
    V = sample_3D_grid(resolution, -1, 1)
    
    pbar = tqdm(range(num_epochs))
    total_loss_history = []
    for epoch in pbar:
        total_loss_val = 0.0
        for P, gt in dataloader:
            P = P.to(DEVICE)
            gt = gt.to(DEVICE)
            
            pred = model(P)
            loss = loss_func(pred, gt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # record loss
        total_loss_val += loss.item()
        total_loss_history.append(total_loss_val)   
        pbar.set_postfix({'total': total_loss_val})

        if (epoch+1) % save_every_epochs == 0:
            plt.figure(0)
            plt.clf()
            plt.semilogy(total_loss_history)
            plt.savefig(folder + "loss_history.png")
            
            torch.save(model, folder + "model_params.pt")

            # batch process voxel points (or it will out of memory)
            pred = torch.zeros_like(V[:,0])
            idx = 0
            for V_mini_batch in torch.split(V, 32**3, 0):
                V_mini_batch = V_mini_batch.unsqueeze(0).to(DEVICE)
                pred_minibatch = model(V_mini_batch).squeeze(0)
                mini_batch_size = pred_minibatch.shape[0]
                pred[idx:idx+mini_batch_size] = pred_minibatch.detach().cpu()
                idx = idx + mini_batch_size
            try:
                write_obj_marching_cubes(folder + "recon.obj", -(pred*2-1))
            except:
                print("failed write_obj_marching_cubes")
                continue

        if (epoch+1) % regenerate_training_data_every_epochs == 0:
            dataloader = single_mesh_dataloader(U,F,num_samples = num_samples, samples_per_batch = samples_per_batch, num_workers = num_cpus)
            
if __name__ == "__main__":
    train()
