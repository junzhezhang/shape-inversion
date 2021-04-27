import numpy as np
import warnings
from numpy.linalg import norm
from scipy.stats import entropy
import torch
import math
from loss import *

import sys

from external.ChamferDistancePytorch.chamfer_python import distChamfer
try: 
    from external.emd.emd_module import emdModule
except:
    print("NOTE: EMD not installed yet")


def MMD_batch(sample_pcs, ref_pcs, batch_size=50, normalize=True, sess=None, verbose=False, use_sqrt=False, use_EMD=False,device=None):
    '''
    compute MMD with CD / EMD between two point sets
    same input and output as minimum_mathing_distance() 
    cuda implementation CD and EMD
    input:
        sample and ref_pcs can be np or tensor
        (full data, like 1000 sample_pcs, and 1000 ref_pcs)
    '''
    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    n_sample, n_pc_points_s, pc_dim_s = sample_pcs.shape
    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible size of point-clouds.')
    
    dist_mat = torch.zeros(n_ref,n_sample)
    
    # np to cuda tensor if start from np
    if isinstance(sample_pcs, np.ndarray):
        ref_pcs = torch.from_numpy(ref_pcs).cuda()
        sample_pcs = torch.from_numpy(sample_pcs).cuda()
    
    for r in range(n_ref):  
        for i in range(0,n_sample,batch_size):
            if i+batch_size < n_sample:
                sample_pcd_seg = sample_pcs[i:i+batch_size]
            else:
                sample_pcd_seg = sample_pcs[i:]

            ref_pcd = ref_pcs[r].unsqueeze(0)
            ref_pcd_e = ref_pcd.expand(sample_pcd_seg.shape[0],n_pc_points,pc_dim)
            
            if use_EMD:
                # EMD
                # ref: https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd
                emd = emdModule()
                dists, assigment = emd(sample_pcd_seg,ref_pcd_e, 0.005, 50)
                dist = dists.mean(dim=1)
            else:
                # CD
                dist1, dist2 , _, _ = distChamfer(ref_pcd_e, sample_pcd_seg)
                dist = dist1.mean(axis=1) + dist2.mean(axis=1)

            if i+batch_size < n_sample:
                dist_mat[r,i:i+batch_size] = dist
            else:
                dist_mat[r,i:] = dist
    mmd_all, _ = dist_mat.min(dim=1)
    mmd_all = mmd_all.detach().cpu().numpy()
    mmd = np.mean(mmd_all)
    mmd = np.sqrt(mmd)
    return mmd, mmd_all, dist_mat.cpu().numpy()

def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """
    # UHD from MPC: https://github.com/ChrisWu1997/Multimodal-Shape-Completion/blob/master/evaluation/completeness.py
    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist

def accuracy(P_recon, P_gt, thre=0.01):
    """
    ACCURACY
    P_gt: N x 3, np array
    P_recon: N x 3, np array
    """
    npoint = P_recon.shape[0]

    P_recon_here = np.expand_dims(P_recon, axis=1) # N x 1 x 3
    P_recon_here = np.tile(P_recon_here, (1, npoint, 1)) # N x N x 3

    P_gt_here = np.tile(P_gt, (npoint,1)) 
    P_gt_here =  np.reshape(P_gt_here, (npoint, npoint, 3)) # N x N x 3

    dists = np.linalg.norm(P_recon_here - P_gt_here, axis=-1) # N x N x 1
    dists = np.squeeze(dists) # N x N

    min_dists = np.amin(dists, axis=1) # 1 x N

    avg_dist = np.mean(min_dists)

    matched = min_dists[min_dists < thre]
    fraction = matched.shape[0] / npoint
    return fraction, avg_dist

def accuracy_cuda(P_recon, P_gt, thre=0.01):
    """
    cuda version of accuracy
    """
    npoint = P_recon.shape[0]
    if isinstance(P_gt, np.ndarray):
        P_recon = torch.from_numpy(P_recon).cuda().unsqueeze(0)
        P_gt = torch.from_numpy(P_gt).cuda().unsqueeze(0)
    else:
        P_recon = P_recon.unsqueeze(0)
        P_gt = P_gt.unsqueeze(0)
    P_recon_here = P_recon.unsqueeze(2).repeat(1,1,npoint,1)
    P_gt_here = P_gt.unsqueeze(1).repeat(1,npoint,1,1)
    
    dist = P_recon_here.add(-P_gt_here)
    dist_value = torch.norm(dist,dim=3).squeeze(0)

    min_dists, _ = dist_value.min(axis=1)
    avg_dist = min_dists.mean()
    
    matched = min_dists[min_dists < thre]
    fraction = matched.shape[0] / npoint
    return fraction, avg_dist

def completeness(P_recon, P_gt, thre=0.01):
    '''
    COMPLETENESS
    P_gt: N x 3, np array
    P_recon: N x 3, np array
    '''

    npoint = P_recon.shape[0]

    P_gt_here = np.expand_dims(P_gt, axis=1) # N x 1 x 3
    P_gt_here = np.tile(P_gt_here, (1, npoint, 1)) # N x N x 3

    P_recon_here = np.tile(P_recon, (npoint,1)) 
    P_recon_here =  np.reshape(P_recon_here, (npoint, npoint, 3)) # N x N x 3

    dists = np.linalg.norm(P_gt_here - P_recon_here, axis=-1) # N x N x 1
    dists = np.squeeze(dists) # N x N

    min_dists = np.amin(dists, axis=1) # N x 1
    
    avg_min_dist = np.mean(min_dists)

    matched = min_dists[min_dists < thre]
    fraction = matched.shape[0] / npoint
    return fraction, avg_min_dist

def completeness_cuda(P_recon, P_gt, thre=0.01):
    """
    completeness_cuda
    """
    npoint = P_recon.shape[0]
    if isinstance(P_gt, np.ndarray):
        P_recon = torch.from_numpy(P_recon).cuda().unsqueeze(0)
        P_gt = torch.from_numpy(P_gt).cuda().unsqueeze(0)
    else:
        P_recon = P_recon.unsqueeze(0)
        P_gt = P_gt.unsqueeze(0)
    P_recon_here = P_recon.unsqueeze(2).repeat(1,1,npoint,1)
    P_gt_here = P_gt.unsqueeze(1).repeat(1,npoint,1,1)
    dist = P_gt_here.add(-P_recon_here)
    dist_value = torch.norm(dist,dim=3).squeeze(0)

    min_dists, _ = dist_value.min(axis=0)
    avg_dist = min_dists.mean()
    
    matched = min_dists[min_dists < thre]
    fraction = matched.shape[0] / npoint
    return fraction, avg_dist

def compute_F1_score(precision, recall):
    f = 2 * precision * recall / (precision + recall)
    return f

def mutual_distance(pcd_ls):
    
    if isinstance(pcd_ls[0],np.ndarray):
        pcd_ls = [torch.from_numpy(itm).unsqueeze(0) for itm in pcd_ls]
    sum_dist = 0
    for i in range(len(pcd_ls)):
        for j in range(i+1,len(pcd_ls)):
            dist1, dist2 , _, _ = distChamfer(pcd_ls[i], pcd_ls[j])
            dist = dist1.mean(axis=1) + dist2.mean(axis=1)
            sum_dist += dist
    mean_dist = sum_dist * 2 / (len(pcd_ls) - 1)
    return mean_dist.item()*10000
