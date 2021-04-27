import torch
import torch.nn as nn
import torch.optim as optim

from data.CRN_dataset import CRNShapeNet
from model.treegan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd

from arguments import Arguments

import time
import numpy as np
from loss import *
from metrics import *
import os
import os.path as osp
from eval_treegan import checkpoint_eval

class TreeGAN():
    def __init__(self, args):
        self.args = args
        
        ### dataset
        self.data = CRNShapeNet(args)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
        print("Training Dataset : {} prepared.".format(len(self.data)))
        
        ### Model
        self.G = Generator(features=args.G_FEAT, degrees=args.DEGREE, support=args.support,args=self.args).to(args.device)
        self.D = Discriminator(features=args.D_FEAT).to(args.device)             
        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))
        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
       
        ### uniform losses
        if self.args.expansion_penality:
            # MSN
            self.expansion = expansionPenaltyModule()
        if self.args.krepul_loss:
            # PU-net
            self.krepul_loss = kNNRepulsionLoss(k=self.args.krepul_k,n_seeds=self.args.krepul_n_seeds,h=self.args.krepul_h)
        if self.args.knn_loss:
            # PatchVariance
            self.knn_loss = kNNLoss(k=self.args.knn_k,n_seeds=self.args.knn_n_seeds)

        print("Network prepared.")

        # ----------------------------------------------------------------------------------------------------- #
        if len(args.w_train_ls) == 1:
            self.w_train_ls = args.w_train_ls * 4
        else:
            self.w_train_ls = args.w_train_ls


    def run(self, save_ckpt=None, load_ckpt=None):        

        epoch_log = 0
        
        loss_log = {'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())

        metric = {'FPD': []}
        if load_ckpt is not None:
            checkpoint = torch.load(load_ckpt, map_location=self.args.device)
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']

            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_legend = list(loss_log.keys())

            metric['FPD'] = checkpoint['FPD']
            
            print("Checkpoint loaded.")
        # parallel after loading
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)
        
        for epoch in range(epoch_log, self.args.epochs):
            epoch_g_loss = []
            epoch_d_loss = []
            epoch_time = time.time()
            self.w_train = self.w_train_ls[min(3,int(epoch/500))]
            for _iter, data in enumerate(self.dataLoader):
                # Start Time
                start_time = time.time()
                point, _, _ = data 
                point = point.to(self.args.device)

                # -------------------- Discriminator -------------------- #
                tic = time.time()
                for d_iter in range(self.args.D_iter):
                    self.D.zero_grad()
                    z = torch.randn(point.shape[0], 1, 96).to(self.args.device)
                    
                    tree = [z]
                    
                    with torch.no_grad():
                        fake_point = self.G(tree)         
                    
                    D_real, _ = self.D(point)
                    D_fake, _ = self.D(fake_point)
                    gp_loss = self.GP(self.D, point.data, fake_point.data)
                    
                    # compute D loss
                    D_realm = D_real.mean()
                    D_fakem = D_fake.mean()
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    # times weight before backward
                    d_loss*=self.w_train
                    d_loss_gp.backward()
                    self.optimizerD.step()

                loss_log['D_loss'].append(d_loss.item())   
                epoch_d_loss.append(d_loss.item())          
                toc = time.time()
                # ---------------------- Generator ---------------------- #
                self.G.zero_grad()
                
                z = torch.randn(point.shape[0], 1, 96).to(self.args.device)
            
                tree = [z]
                
                fake_point = self.G(tree)
                G_fake, _ = self.D(fake_point)
                
                G_fakem = G_fake.mean()
                g_loss = -G_fakem
                if self.args.expansion_penality:
                    dist, _, mean_mst_dis = self.expansion(fake_point,self.args.expan_primitive_size,self.args.expan_alpha)
                    expansion = torch.mean(dist)
                    g_loss = -G_fakem + self.args.expan_scalar * expansion
                if self.args.krepul_loss: 
                    krepul_loss = self.krepul_loss(fake_point)
                    g_loss = -G_fakem + self.args.krepul_scalar * krepul_loss
                if self.args.knn_loss:
                    knn_loss = self.knn_loss(fake_point)
                    g_loss = -G_fakem + self.args.knn_scalar * knn_loss
    
                g_loss*=self.w_train
                g_loss.backward()
                self.optimizerG.step()

                loss_log['G_loss'].append(g_loss.item())
                epoch_g_loss.append(g_loss.item())
                tac = time.time()
                # --------------------- Visualization -------------------- #
                verbose = None
                if verbose is not None:
                    print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                        "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                        "[ G_Loss ] ", "{: 7.6f}".format(g_loss), 
                        "[ Time ] ", "{:4.2f}s".format(time.time()-start_time),
                        "{:4.2f}s".format(toc-tic),
                        "{:4.2f}s".format(tac-toc))

            # ---------------- Epoch everage loss   --------------- #
            d_loss_mean = np.array(epoch_d_loss).mean()
            g_loss_mean = np.array(epoch_g_loss).mean()
            
            print("[Epoch] ", "{:3}".format(epoch),
                "[ D_Loss ] ", "{: 7.6f}".format(d_loss_mean), 
                "[ G_Loss ] ", "{: 7.6f}".format(g_loss_mean), 
                "[ Time ] ", "{:.2f}s".format(time.time()-epoch_time))
            epoch_time = time.time()

            ### call abstracted eval, which includes FPD
            if self.args.eval_every_n_epoch > 0:
                if epoch % self.args.eval_every_n_epoch == 0 :
                    checkpoint_eval(self.G, self.args.device, n_samples=5000, batch_size=100,conditional=False, ratio='even', FPD_path=self.args.FPD_path,class_choices=self.args.class_choice)

            # ---------------------- Save checkpoint --------------------- #
            if epoch % self.args.save_every_n_epoch == 0 and not save_ckpt == None:
                if len(args.class_choice) == 1:
                    class_name = args.class_choice[0]
                else:
                    class_name = 'multi' 
                torch.save({
                        'epoch': epoch,
                        'D_state_dict': self.D.module.state_dict(),
                        'G_state_dict': self.G.module.state_dict(),
                        'D_loss': loss_log['D_loss'],
                        'G_loss': loss_log['G_loss'],
                        'FPD': metric['FPD']
                }, save_ckpt+str(epoch)+'_'+class_name+'.pt')
                               

if __name__ == '__main__':

    args = Arguments(stage='pretrain').parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    if not osp.isdir('./pretrain_checkpoints'):
        os.mkdir('./pretrain_checkpoints')
        print('pretrain_checkpoints parent directory created.')
    
    if not osp.isdir(args.ckpt_path):
        os.mkdir(args.ckpt_path)

    SAVE_CHECKPOINT = args.ckpt_path + args.ckpt_save if args.ckpt_save is not None else None
    LOAD_CHECKPOINT = args.ckpt_load if args.ckpt_load is not None else None
    # print(args)
    model = TreeGAN(args)
    model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT)
