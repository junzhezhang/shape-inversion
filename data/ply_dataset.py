from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import h5py
import random
from tqdm import tqdm
import pickle
import h5py
import glob
from utils.io import read_ply_xyz, read_ply_from_file_list
from utils.pc_transform import swap_axis


def get_stems_from_pickle(test_split_pickle_path):
    """
    get the stem list from a split, given a pickle file
    """
    with open(test_split_pickle_path, 'rb') as f:
        test_list = pickle.load(f)
    stem_ls = []
    for itm in test_list:
        stem, ext = os.path.splitext(itm)
        stem_ls.append(stem)
    return stem_ls

class PlyDataset(data.Dataset):
    """
    datasets that with Ply format
    without GT: MatterPort, ScanNet, KITTI
        Datasets provided by pcl2pcl
    with GT: PartNet, each subdir under args.dataset_path contains 
        the partial shape raw.ply and complete shape ply-2048.txt.
        Dataset provided by MPC

    """
    def __init__(self, args):
        self.dataset = args.dataset
        self.dataset_path = args.dataset_path

        if self.dataset in ['MatterPort', 'ScanNet', 'KITTI']:
            input_pathnames = sorted(glob.glob(self.dataset_path+'/*'))
            input_ls = read_ply_from_file_list(input_pathnames)
            # swap axis as pcl2pcl and ShapeInversion have different canonical pose
            input_ls_swapped = [swap_axis(itm, swap_mode='n210') for itm in input_ls]
            self.input_ls = input_ls_swapped
            self.stems = range(len(self.input_ls))
        elif self.dataset in ['PartNet']:
            pathnames = sorted(glob.glob(self.dataset_path+'/*'))
            basenames = [os.path.basename(itm) for itm in pathnames]

            self.stems = [int(itm) for itm in basenames]

            input_ls = [read_ply_xyz(os.path.join(itm,'raw.ply')) for itm in pathnames]
            gt_ls = [np.loadtxt(os.path.join(itm,'ply-2048.txt'),delimiter=';').astype(np.float32) for itm in pathnames]
 
            # swap axis as multimodal and ShapeInversion have different canonical pose
            self.input_ls = [swap_axis(itm, swap_mode='210') for itm in input_ls]
            self.gt_ls = [swap_axis(itm, swap_mode='210') for itm in gt_ls]
        else:
            raise NotImplementedError
    
    def __getitem__(self, index):
        if self.dataset in ['MatterPort','ScanNet','KITTI']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            return (input_pcd, stem)
        elif self.dataset  in ['PartNet']:
            stem = self.stems[index]
            input_pcd = self.input_ls[index]
            gt_pcd = self.gt_ls[index]
            return (gt_pcd, input_pcd, stem)
    
    def __len__(self):
        return len(self.input_ls)  