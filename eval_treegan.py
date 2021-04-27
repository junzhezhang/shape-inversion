import torch
import torch.nn as nn
import torch.optim as optim

from data.CRN_dataset import CRNShapeNet
from model.treegan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd, calculate_activation_statistics

from metrics import *
from loss import *

from evaluation.pointnet import PointNetCls
from math import ceil
import argparse
import time
import numpy as np
import time
import os.path as osp
import os
import copy
from utils.common_utils import *
from arguments import Arguments

def save_pcs_to_txt(save_dir, fake_pcs):
    """
    save pcds into txt files
    """
    sample_size = fake_pcs.shape[0]
    if not osp.isdir(save_dir):
        os.mkdir(save_dir)
    for i in range(sample_size):
        np.savetxt(osp.join(save_dir,str(i)+'.txt'), fake_pcs[i], fmt = "%f;%f;%f")  

def generate_pcs(model_cuda,n_pcs=5000,batch_size=50,device=None):
    """
    generate fake pcs for evaluation
    """
    fake_pcs = torch.Tensor([])
    n_pcs = int(ceil(n_pcs/batch_size) * batch_size)
    n_batches = ceil(n_pcs/batch_size)

    for i in range(n_batches):
        z = torch.randn(batch_size, 1, 96).to(device)
        tree = [z]
        with torch.no_grad():
            sample = model_cuda(tree).cpu()
        fake_pcs = torch.cat((fake_pcs, sample), dim=0)
    
    return fake_pcs

def create_fpd_stats(pcs, pathname_save, device):
    """
    create stats of training data for eval FPD
    """
    PointNet_pretrained_path = './evaluation/cls_model_39.pth'

    model = PointNetCls(k=16).to(device)
    model.load_state_dict(torch.load(PointNet_pretrained_path))
    mu, sigma = calculate_activation_statistics(pcs, model, device=device)
    print (mu.shape, sigma.shape)
    np.savez(pathname_save,m=mu,s=sigma)
    print('fpd stats saved into:', pathname_save)

@timeit
def script_create_fpd_stats(args, data2stats='CRN'):
    """
    create stats of training data for eval FPD, calling create_fpd_stats()
    """    
    if data2stats == 'CRN':
        dataset = CRNShapeNet(args)
        dataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

        pathname_save = './evaluation/pre_statistics_CRN_'+args.class_choice+'.npz' 
    else:
        raise NotImplementedError

    ref_pcs = torch.Tensor([])
    for _iter, data in enumerate(dataLoader):
        point, _, _ = data
        ref_pcs = torch.cat((ref_pcs, point),0)
    
    create_fpd_stats(ref_pcs,pathname_save, args.device)
    

@timeit
def checkpoint_eval(G_net, device, n_samples=5000, batch_size=100,conditional=False, ratio='even', FPD_path=None, class_choices=None):
    """
    an abstraction used during training
    """
    G_net.eval()
    fake_pcs = generate_pcs(G_net,n_pcs=n_samples,batch_size=batch_size,device=device)
    fpd = calculate_fpd(fake_pcs, statistic_save_path=FPD_path, batch_size=100, dims=1808, device=device)
    # print(fpd)
    print('----------------------------------------- Frechet Pointcloud Distance <<< {:.2f} >>>'.format(fpd))

@timeit
def test(args, mode='FPD', verbose=True):
    '''
    args needed: 
        n_classes, pcs to generate, ratio of each class, class to id dict???
        model pth, , points to save, save pth, npz for the class, 
    '''
    G_net = Generator(features=args.G_FEAT, degrees=args.DEGREE, support=args.support,args=args).to(args.device)
    checkpoint = torch.load(args.model_pathname, map_location=args.device)
    G_net.load_state_dict(checkpoint['G_state_dict'])
    G_net.eval()
    fake_pcs = generate_pcs(G_net,n_pcs=args.n_samples,batch_size=args.batch_size,device=args.device)
    if mode == 'save':
        save_pcs_to_txt(args.save_sample_path, fake_pcs)
    elif mode == 'FPD':
        fpd = calculate_fpd(fake_pcs, statistic_save_path=args.FPD_path, batch_size=100, dims=1808, device=args.device)
        if verbose:
            print('-----FPD: {:3.2f}'.format(fpd))
    elif mode == 'MMD':
        use_EMD = True
        batch_size = 50 
        normalize = True
        gt_dataset = CRNShapeNet(args)
        
        dataLoader = torch.utils.data.DataLoader(gt_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=10)
        gt_data = torch.Tensor([])
        for _iter, data in enumerate(dataLoader):
            point, partial, index = data
            gt_data = torch.cat((gt_data,point),0)
        ref_pcs = gt_data.detach().cpu().numpy()
        sample_pcs = fake_pcs.detach().cpu().numpy()

        tic = time.time()
        mmd, matched_dists, dist_mat = MMD_batch(sample_pcs,ref_pcs,batch_size, normalize=normalize, use_EMD=use_EMD,device=args.device)
        toc = time.time()
        if verbose:
            print('-----MMD-EMD: {:5.3f}'.format(mmd*100))


if __name__ == '__main__':
    args = Arguments(stage='eval_treegan').parser().parse_args()
    args.device = torch.device('cuda')

    assert args.eval_treegan_mode in ["MMD", "FPD", "save", "generate_fpd_stats"]
    
    if args.eval_treegan_mode == "generate_fpd_stats":
        script_create_fpd_stats(args)
    else:
        test(args,mode=args.eval_treegan_mode)    