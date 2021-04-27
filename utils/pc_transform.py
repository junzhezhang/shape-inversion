import numpy as np
import torch

"""
this file contains various functions for point cloud transformation,
some of which are not used in the clean version of code,
but feel free to use them if you have different forms of point clouds.
"""


def swap_axis(input_np, swap_mode='n210'):
    """
    swap axis for point clouds with different canonical frame
    e.g., pcl2pcl and MPC
    """
    if swap_mode == '021': 
        output_np = np.stack([input_np[:,0], input_np[:,2], input_np[:,1]],axis=1)
    elif swap_mode == 'n210': 
        output_np = np.stack([input_np[:,2]*(-1), input_np[:,1], input_np[:,0]],axis=1)
    elif swap_mode == '210':
        output_np = np.stack([input_np[:,2], input_np[:,1], input_np[:,0]],axis=1)
    else:
        raise NotImplementedError
        
    return output_np 

def scale_numpy(input_array, range=0.25,ax_wise=True):
    """
    scale point cloud in the form of numpy array
    """
    if ax_wise:
        max_abs = np.max(np.abs(input_array),axis=0)
        d0 = input_array[:,0] * (range/max_abs[0])
        d1 = input_array[:,1] * (range/max_abs[1])
        d2 = input_array[:,2] * (range/max_abs[2])
        scaled_array = np.stack([d0,d1,d2], axis=1)
    else:
        """
        scale all dimension by the same value, ie the max(abs)
        """
        max_abs = np.max(np.abs(input_array))
        scaled_array = input_array * (range/max_abs)
    return scaled_array

def scale_numpy_ls(input_ls, range=0.25):
    """
    calling a list of point clouds
    """
    output_ls = []
    for itm in input_ls:
        output = scale_numpy(itm, range=range)
        output_ls.append(output)
    return output_ls

def shift_numpy(input_array, mode='center',additional_limit=None):
    """
    shift 
    """
    if mode == 'center':
        ax_max = np.max(input_array,axis=0)
        ax_min = np.min(input_array,axis=0)
        ax_center = (ax_max+ax_min)/2
        shifted_np = input_array - ax_center
    elif mode == 'given_some_limit':
        print(additional_limit)
        if additional_limit[0] != 'yl':
            raise NotImplementedError
        ax_max = np.max(input_array,axis=0)
        ax_min = np.min(input_array,axis=0)
        ax_min[1] = additional_limit[1] # addtional step
        ax_center = (ax_max+ax_min)/2
        shifted_np = input_array - ax_center
    else:
        raise NotImplementedError # weighted center, pc_mean
    
    return shifted_np

def shift_np_one_dim(input_array, dim=2):
    max_dim = input_array.max(axis=0)
    min_dim = input_array.min(axis=0)
    mean_dim = (max_dim[dim]+min_dim[dim])/2
    input_array[:,dim] -= mean_dim
    return input_array 

def downsample_numpy(input_np, points=1024,seed=0):
    if input_np.shape[0] <= points:
        return input_np
    else:
        np.random.seed(seed)
        indices = np.array(range(input_np.shape[0]))
        np.random.shuffle(indices)
        input_downsampled = input_np[indices[:points]]
        return input_downsampled

def voxelize(image, n_bins=32, pcd_limit=0.5, threshold=0):
    """
    voxelize a point cloud
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).unsqueeze(0)
    pcd_new = image * n_bins + n_bins * 0.5
    pcd_new = pcd_new.type(torch.int64)
    ls_voxels = pcd_new.squeeze(0).tolist() # 2028 of sublists
    try:
        tuple_voxels = [tuple(itm) for itm in ls_voxels]
    except:
        import pdb; pdb.set_trace()
    mask_dict = {}
    for tuple_voxel in tuple_voxels:
        if tuple_voxel not in mask_dict:
            mask_dict[tuple_voxel] = 1
        else:
            mask_dict[tuple_voxel] += 1
    for voxel, cnt in mask_dict.items():
        if cnt <= threshold:
            del mask_dict[voxel]
    return mask_dict

def return_plot_range(pcd, plot_range):
    """
    return a range of point cloud,
    to plot Fig.3 in the main paper
    """
    pcd_new = []
    x1, x2 = plot_range[0]
    y1, y2 = plot_range[1]
    z1, z2 = plot_range[2]
    for i in range(2048):
        xs = pcd[i,0]
        ys = pcd[i,2]
        zs = pcd[i,1]
        if x1 < xs < x2 and y1 < ys < y2 and z1 < zs < z2:
            pcd_new.append(pcd[i])
    pcd_new = np.stack(pcd_new)
    return pcd_new

def reverse_normalize(pc, pc_CRN):
    """ 
    scale up by m and relocate
    """
    m = np.max(np.sqrt(np.sum(pc_CRN**2, axis=1)))
    pc = pc * m
    centroid = np.mean(pc_CRN, axis=0)
    pc = pc + centroid

    return pc

def remove_zeros(partial):
    """
    remove zeros (masked) from a point cloud
    """
    if isinstance(partial, np.ndarray):
        partial = torch.from_numpy(partial)
    norm = torch.norm(partial,dim=1)
    idx =  torch.where(norm > 0)
    partial = partial[idx[0]]
    return partial.numpy()

def retrieve_region(pcd, retrieve_range):
    """
    retrieve a range
    input: np.array (N,3)
    """
    x1, x2 = retrieve_range[0]
    y1, y2 = retrieve_range[1]
    z1, z2 = retrieve_range[2]
    points = []
    for i in range(pcd.shape[0]):
        xs = pcd[i,0]
        ys = pcd[i,2]
        zs = pcd[i,1]
        if x1 < xs < x2 and y1 < ys < y2 and z1 < zs < z2:
            points.append(pcd[i])
    new_pcd = np.stack(points)
    print('new_pcd shape',new_pcd.shape)
    return new_pcd