import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from utils.common_utils import *

class Arguments:
    def __init__(self, stage='pretrain'):
        self._parser = argparse.ArgumentParser(description='Arguments for pretain|inversion|eval_treegan|eval_completion.')
        
        if stage == 'eval_completion':
            self.add_eval_completion_args()
        else:
            self.add_common_args()
            if stage == 'pretrain':
                self.add_pretrain_args()
            elif stage == 'inversion':
                self.add_inversion_args()
            elif stage == 'eval_treegan':
                self.add_eval_treegan_args()

    def add_common_args(self):
        ### data related
        self._parser.add_argument('--class_choice', type=str, default='chair', help='plane|cabinet|car|chair|lamp|couch|table|watercraft')
        self._parser.add_argument('--dataset', type=str, default='CRN', help='CRN|MatterPort|ScanNet|KITTI|PartNet|PFNet')
        self._parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path is required')
        self._parser.add_argument('--split', type=str, default='test', help='NOTE: train if pretrain and generate_fpd_stats; test otherwise')
        
        ### TreeGAN architecture related
        self._parser.add_argument('--DEGREE', type=int, default=[1,  2,   2,   2,   2,   2,   64], nargs='+', help='Upsample degrees for generator.')
        self._parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+', help='Features for generator.')
        self._parser.add_argument('--D_FEAT', type=int, default=[3, 64,  128, 256, 256, 512], nargs='+', help='Features for discriminator.')
        self._parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
        self._parser.add_argument('--loop_non_linear', default=False, type=lambda x: (str(x).lower() == 'true'))
        
        ### others
        self._parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')
        self._parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
        self._parser.add_argument('--ckpt_load', type=str, default='pretrained_models/chair.pt', help='Checkpoint name to load. (default:None)')
    
    def add_pretrain_args(self):
        ### general training related
        self._parser.add_argument('--batch_size', type=int, default=128, help='128 for cabinet, lamp, sofa, and boat due to smaller amounts; you can set up to 512 for plane, car, chair, and table')
        self._parser.add_argument('--epochs', type=int, default=2000, help='Integer value for epochs.')
        self._parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')
        self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        self._parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
        self._parser.add_argument('--w_train_ls', type=float, default=[1], nargs='+', help='train loss weightage')

        ### uniform losses related
        # PatchVariance
        self._parser.add_argument('--knn_loss', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--knn_k', type=int, default=30)
        self._parser.add_argument('--knn_n_seeds', type=int, default=100)
        self._parser.add_argument('--knn_scalar', type=float, default=0.2)
        # PU-Net's uniform loss
        self._parser.add_argument('--krepul_loss', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--krepul_k', type=int, default=10)
        self._parser.add_argument('--krepul_n_seeds', type=int, default=20)
        self._parser.add_argument('--krepul_scalar', type=float, default=1)
        self._parser.add_argument('--krepul_h', type=float, default=0.01)
        # MSN's Expansion-Penalty
        self._parser.add_argument('--expansion_penality', default=False, type=lambda x: (str(x).lower() == 'true'))
        self._parser.add_argument('--expan_primitive_size', type=int, default=64)
        self._parser.add_argument('--expan_alpha', type=float, default=1.5)
        self._parser.add_argument('--expan_scalar', type=float, default=0.1)

        ### ohters
        self._parser.add_argument('--ckpt_path', type=str, default='./pretrain_checkpoints/chair', help='Checkpoint path.')
        self._parser.add_argument('--ckpt_save', type=str, default='tree_ckpt_', help='Checkpoint name to save.')
        self._parser.add_argument('--eval_every_n_epoch', type=int, default=10, help='0 means never eval')
        self._parser.add_argument('--save_every_n_epoch', type=int, default=10, help='save models every n epochs')
     
    def add_inversion_args(self):
        
        ### loss related
        self._parser.add_argument('--w_nll', type=float, default=0.001, help='Weight for the negative log-likelihood loss (default: %(default)s)')
        self._parser.add_argument('--p2f_chamfer', action='store_true', default=False, help='partial to full chamfer distance')
        self._parser.add_argument('--p2f_feature', action='store_true', default=False, help='partial to full feature distance')
        self._parser.add_argument('--w_D_loss', type=float, default=[0.1], nargs='+', help='Discriminator feature loss weight (default: %(default)s)')
        self._parser.add_argument('--directed_hausdorff', action='store_true', default=False, help='directed_hausdorff loss during inversion')
        self._parser.add_argument('--w_directed_hausdorff_loss', type=float, default=1)

        ### mask related
        self._parser.add_argument(
            '--mask_type', type=str, default='none',
            help='none|knn_hole|ball_hole|voxel_mask|tau_mask|k_mask; for reconstruction, jittering, morphing, use none; the proposed for shape completion is k_mask')
        self._parser.add_argument('--k_mask_k', type=int, default=[5,5,5,5], nargs='+', help='the k value for k_mask, i.e., top k to keep')
        self._parser.add_argument('--voxel_bins', type=int, default=32, help='number of bins for voxel mask')
        self._parser.add_argument('--surrounding', type=int, default=0, help='< n surroundings, for the surrounding voxels to be masked as 0, for mask v2' )
        self._parser.add_argument('--tau_mask_dist', type=float, default=[0.01,0.01,0.01,0.01], nargs='+', help='tau for tau_mask')
        self._parser.add_argument('--hole_radius', type=float, default=0.35, help='radius of the single hole, ball hole')
        self._parser.add_argument('--hole_k', type=int, default=500, help='k of knn ball hole')
        self._parser.add_argument('--hole_n', type=int, default=1, help='n holes for knn hole or ball hole')
        self._parser.add_argument('--masking_option', type=str, default="element_product", help='keep zeros with element_prodcut or remove zero with indexing')
        
        ### inversion mode related
        self._parser.add_argument('--inversion_mode', type=str, default='completion', help='reconstruction|completion|jittering|morphing|diversity|ball_hole_diversity|simulate_pfnet')
        ### diversity
        self._parser.add_argument('--n_z_candidates', type=int, default=50, help='number of z candidates prior to FPS, based on partial-to-full CD')
        self._parser.add_argument('--n_outputs', type=int, default=10, help='the number of complete outputs for a given partial shape')

        ### other GAN inversion related
        self._parser.add_argument('--random_G', action='store_true', default=False, help='Use randomly initialized generator? (default: %(default)s)')
        self._parser.add_argument('--select_num', type=int, default=500, help='Number of point clouds pool to select from (default: %(default)s)')
        self._parser.add_argument('--sample_std', type=float, default=1.0, help='Std of the gaussian distribution used for sampling (default: %(default)s)')
        self._parser.add_argument('--iterations', type=int, default=[200, 200, 200, 200], nargs='+', 
            help='For bulk structures, i.e., car, couch, cabinet, and plane, each sub-stage consists of 30 iterations; \
            for thin structures, i.e., chair, lamp, table, and boat, each sub-stage consists of 200 iterations.') 
        self._parser.add_argument('--G_lrs', type=float, default=[2e-7, 1e-6, 1e-6, 2e-7], nargs='+', help='Learning rate steps of Generator')
        self._parser.add_argument('--z_lrs', type=float, default=[1e-2, 1e-4, 1e-4, 1e-6], nargs='+', help='Learning rate steps of latent code z')
        self._parser.add_argument('--warm_up', type=int, default=0, help='Number of warmup iterations (default: %(default)s)')
        self._parser.add_argument('--update_G_stages', type=str2bool, default=[1, 1, 1, 1], nargs='+', help='update_G, control at stage')
        self._parser.add_argument('--progressive_finetune', action='store_true', default=False, help='progressive finetune at each stage')
        self._parser.add_argument('--init_by_p2f_chamfer', action='store_true', default=False, help='init_by_p2f_chamfer instead of D feature distance')
        self._parser.add_argument('--early_stopping', action='store_true', default=False, help='early stopping')
        self._parser.add_argument('--stop_cd', type=float, default=0.0005, help='CD threshold for stopping training (default: %(default)s)')
        self._parser.add_argument('--target_downsample_method', default="", type=str, help='FPS: can optionally downsample via Farthest Point Sampling')
        self._parser.add_argument('--target_downsample_size', default=1024, type=int, help='downsample target to what number by FPS')
        
        ### others
        self._parser.add_argument('--save_inversion_path', default='', help='directory to save generated point clouds')   
        self._parser.add_argument('--dist', action='store_true', default=False, help='Train with distributed implementation (default: %(default)s)')
        self._parser.add_argument('--port', type=str, default='12345', help='Port id for distributed training (default: %(default)s)') 
        self._parser.add_argument('--visualize', action='store_true', default=False, help='')

    def add_eval_completion_args(self):
        self._parser.add_argument('--eval_with_GT', type=str2bool, default=0, help='if eval on real scans, choose false') 
        self._parser.add_argument('--saved_results_path', type=str, required=True, help='path of saved_results for evaluation') 

    def add_eval_treegan_args(self):
        self._parser.add_argument('--eval_treegan_mode', type=str, default='FPD', help='MMD|FPD|save|generate_fpd_stats') 
        self._parser.add_argument('--save_sample_path',required=True, help='dir to save generated point clouds')   
        self._parser.add_argument('--model_pathname', type=str, required=True, help='pathname of the model to evaluate')
        self._parser.add_argument('--batch_size', type=int, default=50, help='Integer value for batch size.') 
        self._parser.add_argument('--n_samples', type=int, default=5000, help='number for points to be generated by the G')
    
    def parser(self):
        return self._parser
    


    
   


