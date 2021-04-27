from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))
from utils.io import read_txt_xyz

def draw_any_set(flag_list, pcd_list, output_dir, output_stem, layout=None,apply_ax_limit=True,ax_limit=0.5,size=1,colors=None,axis_off=False,format='.png',figuresize=None,wspace=None,hspace=None,set_title=True,show=False):
    """
    flexibly draw a list of point clouds 
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    ax_min = 0
    ax_max = 0
    pcd_np_list = []
    for pcd in pcd_list:
        if isinstance(pcd,np.ndarray):
            pcd = torch.from_numpy(pcd)
        if pcd.shape[0] == 1:
            pcd.squeeze_(0)
        pcd = pcd.detach().cpu().numpy()
        pcd_np_list.append(pcd)
        ax_min = min(ax_min, np.min(pcd))
        ax_max = max(ax_max, np.max(pcd))
    # in case the generated points has a larger range
    # ax_limit = min(max(abs(ax_min),ax_max) * 1.05, 0.5)  

    if layout == None:
        row = 1
        col = len(pcd_np_list)
        fig = plt.figure(figsize=(len(pcd_list)*4, 4))
    else:
        row, col = layout
    if figuresize is None:
        fig = plt.figure(figsize=(col*4, row*4))
    else:
        fig = plt.figure(figsize=figuresize)

    for i in range(len(pcd_np_list)):
        pcd = pcd_np_list[i]
        ax = fig.add_subplot(row,col,i+1,projection='3d')
        if colors is None:
            ax.scatter(pcd[:,0],pcd[:,2],pcd[:,1],s=size,label=flag_list[i])
        elif colors[i] is None:
            ax.scatter(pcd[:,0],pcd[:,2],pcd[:,1],s=size,label=flag_list[i])
        else:
            ax.scatter(pcd[:,0],pcd[:,2],pcd[:,1],s=size,label=flag_list[i],color=colors[i])
        if apply_ax_limit:
            ax.set_xlim([-ax_limit,ax_limit])
            ax.set_ylim([-ax_limit,ax_limit])
            ax.set_zlim([-ax_limit,ax_limit ])
        if set_title:
            ax.set_title(flag_list[i])
        if axis_off:
            plt.axis('off')
        
    if wspace is not None or hspace is not None:
        plt.subplots_adjust(wspace=wspace,hspace=hspace)

    if show:
        plt.show()
    else:
        output_f = os.path.join(output_dir,output_stem+format)
        plt.savefig(output_f)

