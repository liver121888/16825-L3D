import argparse
import os
import time
import numpy as np
import math
import torch
import pytorch3d
import mcubes
import imageio

# from pytorch3d.utils import ico_sphere
from r2n2_custom import R2N2
# from pytorch3d.ops import sample_points_from_meshes
# from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import Rotate, axis_angle_to_matrix
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.utils import ico_sphere

import utils_vox
import dataset_location
from fit_data import fit_voxel, fit_pointcloud, fit_mesh
from utils import get_mesh_renderer, get_points_renderer
from train_model import train_model

def render_3d(data, isMesh, output_path, args, dist=1):
    num_views = 36
    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=dist,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=args.device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=args.device)

    if isMesh:
        renderer = get_mesh_renderer(image_size=256)
    else:
        renderer = get_points_renderer(image_size=256)
    rend = renderer(data.extend(num_views), cameras=cameras, lights=lights)

    # fig = plot_scene({
    #     "figure": {
    #         "Mesh": data,
    #         "Camera": cameras,
    #     }
    # })
    # fig.show()

    my_images = (rend[:, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    imageio.mimsave(output_path, list(my_images), duration=1000//15, loop=0)


output_prefix = "output/"
if not os.path.exists(output_prefix):
    os.makedirs(output_prefix)


# 16-825 Assignment 2: Single View to 3D

## 1. Exploring loss functions
class Q1Args:
    lr = 4e-4
    max_iter = 100
    # max_iter = 100000
    n_points = 5000
    w_chamfer = 1.0
    w_smooth = 0.1
    device = 'cuda'

q1_args = Q1Args()

r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True)
feed = r2n2_dataset[0]

feed_cuda = {}
for k in feed:
    if torch.is_tensor(feed[k]):
        feed_cuda[k] = feed[k].to(q1_args.device).float()

### 1.1. Fitting a voxel grid (5 points)
voxels_src = torch.rand(feed_cuda['voxels'].shape,requires_grad=True, device=q1_args.device)
voxel_coords = feed_cuda['voxel_coords'].unsqueeze(0)
voxels_tgt = feed_cuda['voxels']
fit_voxel(voxels_src, voxels_tgt, q1_args)

# vox to mesh
mesh = pytorch3d.ops.cubify(voxels_tgt, device=voxels_tgt.device, thresh=0.5)
vertices, faces = mesh.verts_list()[0], mesh.faces_list()[0]
vertices = vertices.unsqueeze(0)
faces = faces.unsqueeze(0)
textures = torch.ones_like(vertices)
textures = textures * torch.tensor([0.7, 0.7, 1]).to(q1_args.device)
print(vertices.shape, faces.shape, textures.shape)
mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=pytorch3d.renderer.TexturesVertex(textures))
render_3d(mesh, True, output_prefix + 'voxel_fit_tgt.gif', q1_args, dist=4.0)

# vox to mesh
mesh = pytorch3d.ops.cubify(voxels_src, device=voxels_src.device, thresh=0.5)
vertices, faces = mesh.verts_list()[0], mesh.faces_list()[0]
vertices = vertices.unsqueeze(0)
faces = faces.unsqueeze(0)
textures = torch.ones_like(vertices)
textures = textures * torch.tensor([0.7, 0.7, 1]).to(q1_args.device)
print(vertices.shape, faces.shape, textures.shape)
mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=pytorch3d.renderer.TexturesVertex(textures))
render_3d(mesh, True, output_prefix + 'voxel_fit_src.gif', q1_args, dist=4.0)


### 1.2. Fitting a point cloud (5 points)
pointclouds_src = torch.randn([1,q1_args.n_points,3],requires_grad=True, device=q1_args.device)
mesh_tgt = pytorch3d.structures.Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])
pointclouds_tgt = sample_points_from_meshes(mesh_tgt, q1_args.n_points)
fit_pointcloud(pointclouds_src, pointclouds_tgt, q1_args)

r = torch.ones(pointclouds_src.shape[-2:])
r = (r * torch.tensor([0.7, 0.7, 1])).unsqueeze(0).to(q1_args.device)
print(pointclouds_src.shape, r.shape)
point_cloud_tgt = pytorch3d.structures.Pointclouds(points=pointclouds_tgt, features=r).detach()
point_cloud = pytorch3d.structures.Pointclouds(points=pointclouds_src, features=r).detach()
render_3d(point_cloud_tgt, False, output_prefix + 'point_cloud_fit_tgt.gif', q1_args, dist=2.0)
render_3d(point_cloud, False, output_prefix + 'point_cloud_fit.gif', q1_args, dist=2.0)

### 1.3. Fitting a mesh (5 points)
mesh_src = ico_sphere(4, q1_args.device)
mesh_tgt = pytorch3d.structures.Meshes(verts=[feed_cuda['verts']], faces=[feed_cuda['faces']])

# fitting
fit_mesh(mesh_src, mesh_tgt, q1_args)    


vertices, faces = mesh_tgt.verts_list()[0], mesh_tgt.faces_list()[0]
vertices = vertices.unsqueeze(0)
faces = faces.unsqueeze(0)
textures = torch.ones_like(vertices)
textures = textures * torch.tensor([0.7, 0.7, 1]).to(q1_args.device)
# print(vertices.shape, faces.shape, textures.shape)
mesh_tgt = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=pytorch3d.renderer.TexturesVertex(textures)).detach()

vertices, faces = mesh_src.verts_list()[0], mesh_src.faces_list()[0]
vertices = vertices.unsqueeze(0)
faces = faces.unsqueeze(0)
textures = torch.ones_like(vertices)
textures = textures * torch.tensor([0.7, 0.7, 1]).to(q1_args.device)
# print(vertices.shape, faces.shape, textures.shape)
mesh_src = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=pytorch3d.renderer.TexturesVertex(textures)).detach()

render_3d(mesh_tgt, True, output_prefix + 'mesh_tgt.gif', q1_args, dist=1.2)
render_3d(mesh_src, True, output_prefix + 'mesh_fit.gif', q1_args, dist=1.2)


## 2. Reconstructing 3D from single view

class Q2Args:
    arch="resnet18"
    lr=4e-4
    # max_iter=100
    max_iter=100000
    batch_size=32
    num_workers=4
    n_points=1000
    w_chamfer=1.0
    w_smooth=0.1
    save_freq=2000
    # save_freq=20
    load_checkpoint=True
    device="cuda"
    load_feat=True
    # ["vox", "point", "mesh"]
    type="vox"

q2_args = Q2Args()

### 2.1. Image to voxel grid (20 points)
train_model(q2_args)

### 2.2. Image to point cloud (20 points)
q2_args.type = "point"
train_model(q2_args)

### 2.3. Image to mesh (20 points)
q2_args.type = "mesh"
train_model(q2_args)
