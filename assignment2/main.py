import argparse
import os
import time

# from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
# from pytorch3d.ops import sample_points_from_meshes
# from pytorch3d.structures import Meshes
import dataset_location
import torch

from fit_data import fit_voxel, fit_pointcloud, fit_mesh

# 16-825 Assignment 2: Single View to 3D

## 1. Exploring loss functions

parser = argparse.ArgumentParser(description='Model Fit', add_help=False)
parser.add_argument('--lr', default=4e-4, type=float)
parser.add_argument('--max_iter', default=100000, type=int)
parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
parser.add_argument('--n_points', default=5000, type=int)
parser.add_argument('--w_chamfer', default=1.0, type=float)
parser.add_argument('--w_smooth', default=0.1, type=float)
parser.add_argument('--device', default='cuda', type=str) 
args = parser.parse_args()

r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)
feed = r2n2_dataset[0]

feed_cuda = {}
for k in feed:
    if torch.is_tensor(feed[k]):
        feed_cuda[k] = feed[k].to(args.device).float()

### 1.1. Fitting a voxel grid (5 points)
voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=args.device)
voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
voxels_tgt = feed_cuda['voxels']
fit_voxel(voxels_src, voxels_tgt, args)

### 1.2. Fitting a point cloud (5 points)
# fit_pointcloud
### 1.3. Fitting a mesh (5 points)

## 2. Reconstructing 3D from single view
### 2.1. Image to voxel grid (20 points)

