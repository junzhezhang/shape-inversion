

from plyfile import PlyData, PlyElement
import os
import numpy as np
from tqdm import tqdm
import shutil


def read_ply_xyz(filename):
    """ read XYZ point cloud from filename PLY file """
    if not os.path.isfile(filename):
        print(filename)
        assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
    return vertices

def read_all_ply_under_dir(dir):
    '''
    read all .ply under a directory
    return a list of arrays
    '''
    all_filenames = os.listdir(dir)
    ply_filenames = []
    for f in all_filenames:
        if f.endswith('.ply'):
            ply_filenames.append(os.path.join(dir, f))
    ply_filenames.sort()
    
    point_clouds = []
    stems = []
    for ply_f in tqdm(ply_filenames):
        pc = read_ply_xyz(ply_f)
        point_clouds.append(pc)
        
        basename = os.path.basename(ply_f)
        stem, ext = os.path.splitext(basename)
        stems.append(stem)
    
    return point_clouds, stems

def read_ply_from_file_list(file_list):
    '''
    read all .ply from a list
    return a list of numpy array
    '''
    point_clouds = []
    for ply_f in tqdm(file_list):
        if not os.path.isfile(ply_f):
            print('Warning: skipping. ', ply_f)
            continue
        pc = read_ply_xyz(ply_f)
        point_clouds.append(pc)
    
    return point_clouds

def export_ply(pc, filename):
    """
    export .ply from a point cloud
    """
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
    ply_out.write(filename)

def copytree(src, dst, symlinks=False, ignore=None):
    # ref: https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def read_txt_xyz(pathname):
    """
    read a point cloud from txt file
    """
    try:
        pcd = np.loadtxt(pathname,delimiter=';').astype(np.float32)  
    except:
        pcd = np.loadtxt(pathname,delimiter=',').astype(np.float32) 
    return pcd

def export_pcd_to_txt(pcd,output_dir,stem):
    """
    export .txt from a point cloud
    """
    if not isinstance(pcd,np.ndarray):
        pcd = pcd.detach().cpu().numpy()
    np.savetxt(os.path.join(output_dir,str(stem)+'.txt'), pcd, fmt = "%f;%f;%f")  