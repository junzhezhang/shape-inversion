import os
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from data.CRN_dataset import CRNShapeNet
from data.ply_dataset import PlyDataset


from arguments import Arguments

from utils.pc_transform import voxelize
from utils.plot import draw_any_set
from utils.common_utils import *
from utils.inversion_dist import *
from loss import *

from shape_inversion import ShapeInversion

from model.treegan_network import Generator, Discriminator
from external.ChamferDistancePytorch.chamfer_python import distChamfer, distChamfer_raw

import random

class Trainer(object):

    def __init__(self, args):
        self.args = args
        
        if self.args.dist:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        self.inversion_mode = args.inversion_mode
        
        save_inversion_dirname = args.save_inversion_path.split('/')
        log_pathname = './logs/'+save_inversion_dirname[-1]+'.txt'
        args.log_pathname = log_pathname

        self.model = ShapeInversion(self.args)
        if self.inversion_mode == 'morphing':
            self.model2 = ShapeInversion(self.args)
            self.model_interp = ShapeInversion(self.args)
        
        if self.args.dataset in ['MatterPort','ScanNet','KITTI','PartNet']:
            dataset = PlyDataset(self.args)
        else: 
            dataset = CRNShapeNet(self.args)
        
        sampler = DistributedSampler(dataset) if self.args.dist else None

        if self.inversion_mode == 'morphing':
            self.dataloader = DataLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                sampler=sampler,
                num_workers=1,
                pin_memory=False)
        else:
            self.dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                sampler=sampler,
                num_workers=1,
                pin_memory=False)
        
    def run(self):
        """
        The framework support the following inversion_mode:
        - completion: complete given partial shapes in the test set
        - reconstruction: reconstruct given complete shapes
        - jittering: change an compelte object into other plausible shapes of different geometries
        - morphing: interpolate between two given complete shapes
        - diversity: output multiple valid complete shapes given a single partial shape
        - ball_hole_diversity: output multiple valid complete shapes, 
            where the partial shapes is randomly made from complete shapes
        - simulate_pfnet: complete partial shapes, where partial shapes are randomly 
            made from complete shapes following PF-Net
        """
        if self.inversion_mode in ['reconstruction', 'completion', 'jittering','simulate_pfnet']:
            self.train()
        elif self.inversion_mode in ['diversity', 'ball_hole_diversity']: 
            self.train_diversity()
        elif self.inversion_mode == 'morphing':
            self.train_morphing() 
        else:
            raise NotImplementedError

    def train(self):
        for i, data in enumerate(self.dataloader):
            tic = time.time()
            if self.args.dataset in ['MatterPort','ScanNet','KITTI']:
                # without gt
                partial, index = data
                gt = None
            else:
                # with gt
                gt, partial, index = data
                gt = gt.squeeze(0).cuda()
                
                ### simulate pfnet ball-holed data
                if self.args.inversion_mode == 'simulate_pfnet':
                    n_removal = 512
                    choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                    chosen = random.sample(choice,1)
                    dist = gt.add(-chosen[0].cuda())
                    dist_val = torch.norm(dist,dim=1)
                    top_dist, idx = torch.topk(dist_val, k=2048-n_removal)
                    partial = gt[idx]
            
            partial = partial.squeeze(0).cuda()
            # reset G for each new input
            self.model.reset_G(pcd_id=index.item())

            # set target and complete shape 
            # for ['reconstruction', 'jittering', 'morphing'], GT is used for reconstruction
            # else, GT is not involved for training
            if partial is None or self.args.inversion_mode in ['reconstruction', 'jittering', 'morphing','ball_hole','knn_hole']:
                self.model.set_target(gt=gt)
            else:
                self.model.set_target(gt=gt, partial=partial)
            
            # initialization
            self.model.select_z(select_y=False)
            # inversion
            self.model.run()
            toc = time.time()
            if self.rank == 0:
                print(i ,'out of',len(self.dataloader),'done in ',int(toc-tic),'s')
            
            if self.args.visualize:
                pcd_list = self.model.checkpoint_pcd
                flag_list = self.model.checkpoint_flags
                output_dir = self.args.save_inversion_path + '_visual'
                if self.args.inversion_mode == 'jittering':
                    output_stem = str(index.item())
                    draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(4,10))
                else:
                    output_stem = str(index.item())
                    draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=(3,4))
        
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<,rank',self.rank,'completed>>>>>>>>>>>>>>>>>>>>>>')

    def train_diversity(self):
        for i, data in enumerate(self.dataloader):
            tic = time.time()
            ### get data
            if self.args.dataset in ['MatterPort','ScanNet','KITTI']:
                # without gt
                partial, index = data
                gt = None
            else:
                # with gt
                gt, partial, index = data
                gt = gt.squeeze(0).cuda()

            if self.args.inversion_mode == 'ball_hole_diversity':
                pcd = gt.unsqueeze(0).clone()
                self.hole_radius = self.args.hole_radius
                self.hole_n = self.args.hole_n
                seeds = farthest_point_sample(pcd, self.hole_n) # shape (B,hole_n)
                self.hole_centers = torch.stack([itm[seed] for itm, seed in zip(pcd,seeds)]) # (B, hole_n, 3)
                flag_map = torch.ones(1,2048,1).cuda()
                pcd_new = pcd.unsqueeze(2).repeat(1,1,self.hole_n,1)
                seeds_new = self.hole_centers.unsqueeze(1).repeat(1,2048,1,1)
                delta = pcd_new.add(-seeds_new) # (B, 2048, hole_n, 3)
                dist_mat = torch.norm(delta,dim=3)
                dist_new = dist_mat.transpose(1,2) # (B, hole_n, 2048)
                for i in range(self.hole_n):
                    dist_per_hole = dist_new[:,i,:].unsqueeze(2)
                    threshold_dist = self.hole_radius
                    flag_map[dist_per_hole <= threshold_dist] = 0
                partial = torch.mul(pcd, flag_map).squeeze(0)
                ### remove zeros
                norm = torch.norm(partial,dim=1)
                idx =  torch.where(norm > 0)
                partial = partial[idx[0]]
                partial = partial.cuda() 
                # print(index.item(), 'partial shape', partial.shape)
            else:
                partial = partial.squeeze(0).cuda()
            
            # reset G for each new input
            self.model.reset_G(pcd_id=index.item())
            # set target and complete shape
            self.model.set_target(gt=gt, partial=partial)
            # search init values of z
            self.model.diversity_search()

            ### fine tuning
            pcd_ls = [gt.unsqueeze(0), partial.unsqueeze(0)]
            flag_ls = ['gt', 'input']
            for ith, z in enumerate(self.model.zs):
                self.model.reset_G(pcd_id=index.item())
                self.model.set_target(gt=gt, partial=partial)
                self.model.z.data = z.data
                self.model.run(ith=ith)
                self.model.xs.append(self.model.x)
                flag_ls.append(str(ith))
                pcd_ls.append(self.model.x)
            
            if self.args.visualize:
                output_stem = str(index.item())
                output_dir = self.args.save_inversion_path + '_visual'
                if self.args.n_outputs <= 10:
                    layout = (3,4)
                elif self.args.n_outputs <= 20:
                    layout = (4,6)
                else:
                    layout = (6,9)
                draw_any_set(flag_ls, pcd_ls, output_dir, output_stem, layout=layout)
                if self.rank == 0:
                    toc = time.time()
                    print(i ,'out of',len(self.dataloader),'done in ',int(toc-tic),'s')
                    tic = time.time()

    def train_morphing(self):
        """
        shape interpolation
        shape pairs are randomly predefined by the dataloader
        """
        for i, data in enumerate(self.dataloader):
            tic = time.time()

            gt, partial, index = data
            gt = gt.cuda()
            partial = partial.cuda()
            # conduct reconstruction of both pcd 
            self.model.reset_G(pcd_id=index[0].item())
            self.model.set_target(gt=gt[0])
            self.model.select_z(select_y=False)
            self.model.run()

            self.model2.reset_G(pcd_id=index[1].item())
            self.model2.set_target(gt=gt[1])
            self.model2.select_z(select_y=False)
            self.model2.run()

            # do interpolation on both z and G
            interpolated_pcd = []
            interpolated_flag = []
            weight1 = self.model.G.state_dict()
            weight2 = self.model2.G.state_dict()
            weight_interp = OrderedDict()
            with torch.no_grad():
                for i in range(11):
                    alpha = i / 10
                    # interpolate between both latent vector and generator weight
                    z_interp = alpha * self.model.z + (1 - alpha) * self.model2.z
                    for k, w1 in weight1.items():
                        w2 = weight2[k]
                        weight_interp[k] = alpha * w1 + (1 - alpha) * w2
                    self.model_interp.G.load_state_dict(weight_interp)
                    x_interp = self.model_interp.G([z_interp])
                    interpolated_pcd.append(x_interp)
                    interpolated_flag.append(str(alpha))
            
            if self.args.visualize:
                pcd_ls = [gt[1]] + interpolated_pcd + [gt[0]]
                flag_ls = ['gt_1'] + interpolated_flag + ['gt_2']
                output_dir = self.args.save_inversion_path + '_visual'
                output_stem = str(index[0].item())+'_'+str(index[1].item())
                draw_any_set(flag_ls, pcd_ls, output_dir, output_stem, layout=(3,6))

                output_dir2 = self.args.save_inversion_path + '_interpolates'
                if not os.path.isdir(output_dir2):
                    os.mkdir(output_dir2)
                for flag, pcd in zip(flag_ls, pcd_ls):
                    pcd = pcd.squeeze(0).detach().cpu().numpy()
                    np.savetxt(os.path.join(output_dir2, output_stem+'_'+flag+'.txt'),pcd,fmt = "%f;%f;%f")
            
            if self.rank == 0:
                toc = time.time()
                print(i ,'out of',len(self.dataloader),'done in ',int(toc-tic),'s')
                tic = time.time()          

if __name__ == "__main__":
    args = Arguments(stage='inversion').parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    
    if not os.path.isdir('./logs/'):
        os.mkdir('./logs/')
    if not os.path.isdir('./saved_results'):
        os.mkdir('./saved_results')
    
    if args.dist:
        rank, world_size = dist_init(args.port)

    trainer = Trainer(args)
    trainer.run()
    
    