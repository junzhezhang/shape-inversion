
import os
import torch
import numpy as np
from external.ChamferDistancePytorch.chamfer_python import distChamfer
import glob
from metrics import *
from loss import *
import h5py
from utils.common_utils import *
from arguments import Arguments

def compute_cd_small_batch(gt, output,batch_size=50):
    """
    compute cd in case n_pcd is large
    """
    n_pcd = gt.shape[0]
    dist = []
    for i in range(0, n_pcd, batch_size):
        last_idx = min(i+batch_size,n_pcd)
        dist1, dist2 , _, _ = distChamfer(gt[i:last_idx], output[i:last_idx])
        cd_loss = dist1.mean(1) + dist2.mean(1)
        dist.append(cd_loss)
    dist_tensor = torch.cat(dist)
    cd_ls = (dist_tensor*10000).cpu().numpy().tolist()
    return cd_ls

def compute_3_metrics(gt, output,thre=0.01):
    """
    compute acc, comp, f1 for a batch
    input can be np or tensor
    return lists
    """
    if not isinstance(gt, np.ndarray):
        gt = gt.detach().cpu().numpy()
        output = output.detach().cpu().numpy()
    
    acc_ls = []
    comp_ls = []
    f1_ls = []
    for i in range(gt.shape[0]):
        acc, avg_dist = accuracy_cuda(output[i],gt[i],thre=thre)
        comp, avg_min_dist = completeness_cuda(output[i],gt[i],thre=thre)
        f1 = compute_F1_score(acc, comp)
        acc_ls.append(acc)
        comp_ls.append(comp)
        f1_ls.append(f1)

    return [itm*100 for itm in acc_ls], [itm*100 for itm in comp_ls], [itm*100 for itm in f1_ls]

def compute_4_metrics(pcn_gt, pcn_output):
    """
    compute cd, acc, comp, f1 for a batch
    """
    if pcn_gt.shape[0] <= 150:
        dist1, dist2 , _, _ = distChamfer(pcn_gt, pcn_output)
        cd = dist1.mean(1)+ dist2.mean(1)
        cd_ls  = (cd*10000).cpu().numpy().tolist()
    else:
        ### compute with small batches:
        batch_size = 50
        gt = pcn_gt
        cd_ls = compute_cd_small_batch(pcn_gt, pcn_output, batch_size=batch_size)
    
    acc_ls, comp_ls, f1_ls = compute_3_metrics(pcn_gt,pcn_output,thre=0.03)

    return cd_ls, acc_ls, comp_ls, f1_ls

def compute_ucd(partial_ls, output_ls):
    """ 
    input two lists (small lists)
    return a single mean
    """
    if isinstance(partial_ls[0],np.ndarray):
        partial_ls = [torch.from_numpy(itm) for itm in partial_ls]
        output_ls = [torch.from_numpy(itm) for itm in output_ls]
    if len(partial_ls) < 100:
        partial = torch.stack(partial_ls).cuda()
        output = torch.stack(output_ls).cuda()
        dist1, dist2 , _, _ = distChamfer(partial, output)
        cd_loss = dist1.mean()*10000
        cd_ls = (dist1.mean(1)*10000).cpu().numpy().tolist()
        return cd_loss.item(), cd_ls
    else:
        batch_size = 50
        n_samples = len(partial_ls)
        n_batches = int(n_samples/batch_size) + min(1, n_samples%batch_size)
        cd_ls = []
        for i in range(n_batches):
            # if i*batch_size
            # print(n_samples, i, i*batch_size)
            partial = torch.stack(partial_ls[i*batch_size:min(n_samples,i*batch_size+batch_size)]).cuda()
            output = torch.stack(output_ls[i*batch_size:min(n_samples,i*batch_size+batch_size)]).cuda()
            dist1, dist2 , _, _ = distChamfer(partial, output)
            cd_loss = dist1.mean(1)*10000
            cd_ls.append(cd_loss)
        cd = torch.cat(cd_ls).mean().item()
        cd_ls = torch.cat(cd_ls).cpu().numpy().tolist()
        return cd, cd_ls

def compute_uhd(partial_ls, output_ls):
    """
    input two lists (small lists)
    return a single mean
    """
    if isinstance(partial_ls[0],np.ndarray):
        partial_ls = [torch.from_numpy(itm) for itm in partial_ls]
        output_ls = [torch.from_numpy(itm) for itm in output_ls]
    partial = torch.stack(partial_ls).cuda()
    output = torch.stack(output_ls).cuda()
    uhd = DirectedHausdorff()

    udh_loss = uhd(partial, output)
    
    return udh_loss.item()

def retrieve_ours_pcs(input_dir, with_input=True, with_gt=True):
    pathnames = sorted(glob.glob(input_dir+"/*"))
    
    ids = set()
    for pathname in pathnames:
        stem, ext = os.path.splitext(os.path.basename(pathname))
        stem_id = stem.split('_')[0]
        ids.add(int(stem_id))
    index_ls = sorted(list(ids))
    ours_output_ls = []
    ours_gt_ls = []
    ours_target_ls = []
    for idx in index_ls:
        output_pathname = os.path.join(input_dir, str(idx)+'_x.txt')
        np_output = np.loadtxt(output_pathname,delimiter=';').astype(np.float32) 
        ours_output_ls.append(torch.from_numpy(np_output))
        
        if with_gt: 
            gt_pathname = os.path.join(input_dir, str(idx)+'_gt.txt')
            np_gt = np.loadtxt(gt_pathname,delimiter=';').astype(np.float32) 
            ours_gt_ls.append(torch.from_numpy(np_gt))
        
        if with_input:
            target_pathname = os.path.join(input_dir, str(idx)+'_target.txt')
            np_target = np.loadtxt(target_pathname,delimiter=';').astype(np.float32) 
            ours_target_ls.append(torch.from_numpy(np_target))
    ours_output = torch.stack(ours_output_ls).cuda()
    
    if with_gt:
        ours_gt = torch.stack(ours_gt_ls).cuda()
    else:
        ours_gt = None
   
    if with_input :
        ours_input = torch.stack(ours_target_ls).cuda()
    else:
        ours_input = None
    return ours_gt, ours_output, ours_input

 
def eval_completion_with_gt(input_dir, cd_verbose=False):
    
    ours_gt, ours_output, _ = retrieve_ours_pcs(input_dir)
    cd_ls, acc_ls, comp_ls, f1_ls = compute_4_metrics(ours_gt, ours_output)
    if cd_verbose:
        dist1, dist2 , _, _ = distChamfer(ours_gt, ours_output)
        cd_loss = dist1.mean() + dist2.mean()
        if cd_verbose:
            cd_pcds = dist1.mean(1)+ dist2.mean(1)
            cd_pcds*=10000
            for i in range(150):
                print(i,int(cd_pcds[i].item()))
    print('cd.mean:',np.mean(cd_ls)) 
    print('acc :',np.mean(acc_ls))
    print('comp:',np.mean(comp_ls))
    print('f1  :',np.mean(f1_ls))


def eval_completion_without_gt(input_dir):
    ### retrieve _x and target
    pathnames = glob.glob(input_dir+"/*")
    ids = set()
    for pathname in pathnames:
        stem, ext = os.path.splitext(os.path.basename(pathname))
        stem_id = stem.split('_')[0]
        ids.add(stem_id)
    index_ls = sorted(list(ids))
    ours_output_ls = []
    ours_target_ls = []
    for idx in index_ls:
        gt_pathname = os.path.join(input_dir, str(idx)+'_target.txt')
        output_pathname = os.path.join(input_dir, str(idx)+'_x.txt')
        np_gt = np.loadtxt(gt_pathname,delimiter=';').astype(np.float32) 
        np_output = np.loadtxt(output_pathname,delimiter=';').astype(np.float32) 
        ours_target_ls.append(torch.from_numpy(np_gt))
        ours_output_ls.append(torch.from_numpy(np_output))
    cd, cd_ls = compute_ucd(ours_target_ls, ours_output_ls)
    uhd = compute_uhd(ours_target_ls, ours_output_ls)
    print('ucd:',cd)
    print('uhd:',uhd)

if __name__ == '__main__':

    args = Arguments(stage='eval_completion').parser().parse_args()
    if args.eval_with_GT:
        eval_completion_with_gt(args.saved_results_path)
    else:
        eval_completion_without_gt(args.saved_results_path)

    
    