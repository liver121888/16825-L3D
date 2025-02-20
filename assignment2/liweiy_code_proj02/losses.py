import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.ops import knn_points

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# implement some loss for binary voxel grids

	# print("voxel_src: ", voxel_src.shape)
	# n = voxel_src.shape[1] * voxel_src.shape[2] * voxel_src.shape[3]
	# loss = 0
	# for i in range(voxel_src.shape[0]):
	# 	vs = voxel_src[i, ...]
	# 	print(vs.shape)
	# 	vt = voxel_tgt[i, ...]
	# 	print(vt.shape)
	# 	loss += -1/n * torch.sum(vs * torch.log(vt) + (1-vs) * torch.log(1-vt))
	# print(loss)

	# b, h, w, d = voxel_src.shape
	# print(voxel_src[0, 0, 0, 0])
	bce_loss = nn.BCEWithLogitsLoss()
	loss = bce_loss(voxel_src, voxel_tgt)
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# implement chamfer loss from scratch

	# print("src/tgt shapes: ",point_cloud_src.shape, " ", point_cloud_tgt.shape)
	
	# reference: mingfenl
	# tgt_n = point_cloud_tgt.shape[1]
	# loss_chamfer = 0.
	# with torch.no_grad():
	# 	point_cloud_src_expaned = point_cloud_src.unsqueeze(2).expand(-1, -1, tgt_n, -1) 
	# 	diff = point_cloud_src_expaned - point_cloud_tgt.unsqueeze(1)
	# 	diff_square = torch.sum(diff**2, dim=-1)

	# 	min_tgt_src_idxs = torch.argmin(diff_square, dim=2)
	# 	min_src_tgt_idxs = torch.argmin(diff_square, dim=1)

	# for x, y, min_idx_tgt, min_idx_src in zip(point_cloud_src, point_cloud_tgt, min_tgt_src_idxs, min_src_tgt_idxs):
	# 	loss_chamfer += torch.sum((x - y[min_idx_tgt])**2) + torch.sum((y - x[min_idx_src])**2)

    dists_st, _, _ = knn_points(
        point_cloud_src, 
        point_cloud_tgt,
        K=1
    )
    dists_ts, _, _ = knn_points(
        point_cloud_tgt, 
        point_cloud_src,
        K=1
    )
    loss_chamfer = torch.mean(torch.mean(dists_st) + torch.mean(dists_ts))
    return loss_chamfer

	# compare: pytorch3d
	# loss_chamfer = chamfer_distance(point_cloud_src, point_cloud_tgt)[0]

	# print(loss_chamfer)
	# return loss_chamfer

def smoothness_loss(mesh_src):
	# implement laplacian smoothening loss

	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian