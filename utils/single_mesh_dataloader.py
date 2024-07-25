import numpy as np
import igl
import torch
from torch.utils.data import TensorDataset, DataLoader
from .to_numpy import to_numpy

def single_mesh_dataloader(V,F,
						   num_samples = 1024 * 100,
						   samples_per_batch = 1024,
						   num_workers = 0,
						   sample_ratio = [0.4, 0.4, 0.2]):
	# This function creates a pytorch dataloader that load 3D points sampled in the space and return the occupancy value of these points wrt a given mesh
	
    func, P = sample_mesh_occupancy(to_numpy(V),to_numpy(F),num_samples, ratio=sample_ratio)
    func = torch.from_numpy(func).type(torch.float)
    P = torch.from_numpy(P).type(torch.float)

    P_func = torch.hstack((P, func[:,None]))
    P_func_batches = torch.split(P_func, samples_per_batch, dim=0)
    P_func_batches = torch.stack(P_func_batches)

    dataset = TensorDataset(*[P_func_batches[:,:,:3], P_func_batches[:,:,-1]])
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=num_workers)
    return dataloader

def sample_mesh_occupancy(V,F,num_samples, \
    ratio = [0.4, 0.4, 0.2], 
    near_surface_displacement = 0.1):
    """
    This function sample points in space and compute their sign values to the mesh. The sampling methods include "uniform" and "importance" sampling. Importance sampling is often recommended. It includes (1) on-surface, (2) near-surface, (3) random sampled points with proportion controlled by "ratio".

    Inputs:
        V: |V|x3 array of vertex locations
        F: |F|x3 array of face indices 
        num_samples: number of points to sample
        method: method for doing sampling (either "importance" or "uniform")
        ratio: 3 dim vector for the proportion for importance samplings [on-surface, near-surface, random]
        near_surface_displacement: displacement vector for near surface sampling.

    Outputs:
        occu: num_samples vector of occupancy values
        P: num_samples x 3 array of sample locations
    
    [WARNING]:
        assume the input mesh is closed 
    """
    # making sure ratio is valid
    ratio = np.array(ratio).astype(np.float32)
    ratio = np.clip(ratio, 0, ratio.max()) # remove negative values
    ratio /= ratio.sum() # normalzie to sum up to one

    # compute face normal (for determining sign)
    FN = face_normals(V,F)

    # perform importance sampling near the surface
    # compute area cumsum 
    FA = face_areas(V,F)
    FA = FA / np.sum(FA)
    FA_cumsum = np.cumsum(FA)

    # sample points on surface 
    num_surface_samples = int(np.round(num_samples * ratio[0]))
    bF = np.searchsorted(FA_cumsum, np.random.rand(num_surface_samples))
    bC = np.random.rand(num_surface_samples, 3)
    bC = bC / np.sum(bC, 1)[:,None]
    P_surf = bC[:,[0]] * V[F[bF,0],:] + bC[:,[1]] * V[F[bF,1],:] + bC[:,[2]] * V[F[bF,2],:]

    # sample points near surface
    num_near_surface_samples = int(np.round(num_samples * ratio[1]))
    bF = np.searchsorted(FA_cumsum, np.random.rand(num_near_surface_samples))
    bC = np.random.rand(num_near_surface_samples, 3)
    bC = bC / np.sum(bC, 1)[:,None]
    perturb = (np.random.rand(num_near_surface_samples, 1) - 0.5) * near_surface_displacement * 2
    P_near_surf = bC[:,[0]] * V[F[bF,0],:] + bC[:,[1]] * V[F[bF,1],:] + bC[:,[2]] * V[F[bF,2],:] + perturb * FN[bF,:]
    P_near_surf = np.clip(P_near_surf, -1, 1) # don't want to sample outside the -1~1 bounding box

    # sample points randomly in R3
    num_random_samples = num_samples - num_near_surface_samples - num_surface_samples
    P_uniform = np.array(np.random.rand(num_random_samples, 3) - 0.5) * 2.0

    # concatenate points
    P = np.concatenate((P_surf, P_near_surf, P_uniform), axis=0)
    np.random.shuffle(P) # shuffle rows

    occu,_,_ = igl.signed_distance(P, V, F)
    idx_outside = (occu > 0)
    occu[occu <= 0] = 1.0
    occu[idx_outside] = 0.0
    return occu, P

def face_normals(V, F, f = None):    
	'''
	computes unit face normal of a triangle mesh

	Inputs:
        V: |V|x3 numpy ndarray of vertex positions
        F: |F|x3 numpy ndarray of face indices
		f: (optional) a subset of face indices 
	
	Outputs:
	    FN: |F|x3 (or |f|x3) numpy ndarray of unit face normal 
	'''
	if np.isscalar(f): # f is a integer
		vec1 = V[F[f,1],:] - V[F[f,0],:]
		vec2 = V[F[f,2],:] - V[F[f,0],:]
		FN = np.cross(vec1, vec2)
		return FN / np.linalg.norm(FN)
	if f is None: # compute all
		vec1 = V[F[:,1],:] - V[F[:,0],:]
		vec2 = V[F[:,2],:] - V[F[:,0],:]
	else: # compute a subset
		vec1 = V[F[f,1],:] - V[F[f,0],:]
		vec2 = V[F[f,2],:] - V[F[f,0],:]
	FN = np.cross(vec1, vec2) 
	return FN / np.sqrt(np.sum(FN**2,1))[:,None]

def face_areas(V, F):
    """
    computes area per face 

    Input:
        V |V|x3 numpy array of vertex positions
        F |F|x3 numpy array of face indices
    Output:
        FA |F| numpy array of face areas
    """
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN = np.cross(vec1, vec2) / 2
    FA = np.sqrt(np.sum(FN**2,1))
    return FA