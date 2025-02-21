# Assignment 2

Liwei Yang, liweiy@andrew.cmu.edu

- [Assignment 2](#assignment-2)
  - [1. Exploring loss functions](#1-exploring-loss-functions)
    - [1.1. Fitting a voxel grid (5 points)](#11-fitting-a-voxel-grid-5-points)
    - [1.2. Fitting a point cloud (5 points)](#12-fitting-a-point-cloud-5-points)
    - [1.3. Fitting a mesh (5 points)](#13-fitting-a-mesh-5-points)
  - [2. Reconstructing 3D from single view](#2-reconstructing-3d-from-single-view)
    - [2.1. Image to voxel grid (20 points)](#21-image-to-voxel-grid-20-points)
    - [2.2. Image to point cloud (20 points)](#22-image-to-point-cloud-20-points)
    - [2.3. Image to mesh (20 points)](#23-image-to-mesh-20-points)
    - [2.4. Quantitative comparisions(10 points)](#24-quantitative-comparisions10-points)
    - [2.5. Analyse effects of hyperparams variations (10 points)](#25-analyse-effects-of-hyperparams-variations-10-points)
    - [2.6. Interpret your model (15 points)](#26-interpret-your-model-15-points)
  - [3. Exploring other architectures / datasets.](#3-exploring-other-architectures--datasets)

## 1. Exploring loss functions

### 1.1. Fitting a voxel grid (5 points)

Optimized voxel grid

![voxel_fit_src](data/voxel_fit_src.gif)

Ground truth voxel grid

![voxel_fit_tgt](data/voxel_fit_tgt.gif)

### 1.2. Fitting a point cloud (5 points)

Optimized point cloud

![point_cloud_fit_src](data/point_cloud_fit_src.gif)

Ground truth point cloud

![point_cloud_fit_tgt](data/point_cloud_fit_tgt.gif)

### 1.3. Fitting a mesh (5 points)

Optimized mesh

![mesh_fit_src](data/mesh_fit_src.gif)

Ground truth mesh

![mesh_fit_tgt](data/mesh_fit_tgt.gif)

## 2. Reconstructing 3D from single view


### 2.1. Image to voxel grid (20 points)

Sample \# 0

![vox_eval_0](data/vox_eval_gt_image_0.png)

Ground truth mesh

![vox_eval_0](data/vox_eval_gt_0.gif)

Predicted vox

![vox_eval_0](data/vox_eval_0.gif)

Sample \# 100

![vox_eval_100](data/vox_eval_gt_image_100.png)

Ground truth mesh

![vox_eval_100](data/vox_eval_gt_100.gif)

Predicted vox

![vox_eval_100](data/vox_eval_100.gif)

Sample \# 400

![vox_eval_400](data/vox_eval_gt_image_400.png)

Ground truth mesh

![vox_eval_400](data/vox_eval_gt_400.gif)

Predicted vox

![vox_eval_400](data/vox_eval_400.gif)

### 2.2. Image to point cloud (20 points)

Sample \# 0

![point_eval_0](data/point_eval_gt_image_0.png)

Ground truth mesh

![point_eval_0](data/point_eval_gt_0.gif)

Predicted point cloud

![point_eval_0](data/point_eval_0.gif)

Sample \# 100

![point_eval_100](data/point_eval_gt_image_100.png)

Ground truth mesh

![point_eval_100](data/point_eval_gt_100.gif)

Predicted point cloud

![point_eval_100](data/point_eval_100.gif)

Sample \# 400

![point_eval_400](data/point_eval_gt_image_400.png)

Ground truth mesh

![point_eval_400](data/point_eval_gt_400.gif)

Predicted point cloud

![point_eval_400](data/point_eval_400.gif)

### 2.3. Image to mesh (20 points)

Sample \# 0

![mesh_eval_0](data/mesh_eval_gt_image_0.png)

Ground truth mesh

![mesh_eval_0](data/mesh_eval_gt_0.gif)

Predicted mesh

![mesh_eval_0](data/mesh_eval_0.gif)

Sample \# 100

![mesh_eval_100](data/mesh_eval_gt_image_100.png)

Ground truth mesh

![mesh_eval_100](data/mesh_eval_gt_100.gif)

Predicted mesh

![mesh_eval_100](data/mesh_eval_100.gif)

Sample \# 400

![mesh_eval_400](data/mesh_eval_gt_image_400.png)

Ground truth mesh

![mesh_eval_400](data/mesh_eval_gt_400.gif)

Predicted mesh

![mesh_eval_400](data/mesh_eval_400.gif)

### 2.4. Quantitative comparisions(10 points)

Voxel evaluation

![eval_vox](data/eval_vox.png)

Point cloud evaluation

![eval_point](data/eval_point.png)

Mesh evaluation

![eval_mesh](data/eval_mesh.png)

From the result, we can see point cloud model achieve the best F1 score, and voxel model has the worst performance.

One reasone behind this might be some of the chair legs, or other thin features are too small to occupy one voxel. thus from sample \#400 we can see the chair is missing the legs. For mesh model, the data structure is in natrual more complex than point cloud, thus the thin legs are also often only represented by a thin face. Point cloud model benefits from the simple data structure and thus has the best performace in terms of F1 score. 

### 2.5. Analyse effects of hyperparams variations (10 points)

I change the n_points of point cloud model to 1500 points. The evaluation result is as below:

![eval_point_1500](data/eval_point_1500.png)

Sample \# 400

![mesh_eval_400](data/mesh_eval_gt_image_400.png)

Ground truth mesh

![mesh_eval_400](data/mesh_eval_gt_400.gif)

1000 points

![point_eval_400](data/point_eval_400.gif)

1500 points

![point_eval_400_1500](data/point_eval_400_1500.gif)

We can see the back panel becomes denser and represent the mesh ground truth better. The F1 score is also generally higher when compare to the same threshold of 1000 points. Adding more points should amplify this difference more.

### 2.6. Interpret your model (15 points)

I visulize the training process of mesh model

iter 0

![mesh_traintrain_0](data/mesh_traintrain_0.gif)

iter 50

![mesh_traintrain_50](data/mesh_traintrain_50.gif)

iter 150

![mesh_traintrain_150](data/mesh_traintrain_150.gif)

iter 250

![mesh_traintrain_250](data/mesh_traintrain_250.gif)

From iter 0, we can see the isosphere is still big and the cameras are within the isosphere. As we train more iters, the mesh gradually forms a general chair sturcture.

## 3. Exploring other architectures / datasets.

![eval_point_full](data/eval_point_full.png)

Sample \#40

![point_eval_f_gt_40](data/point_eval_f_gt_image_40.png)

Model prediction

![40](data/40_eval.gif)

Ground truth mesh

![point_eval_f_gt_40](data/40_gt_eval.gif)


Sample \#460

![point_eval_f_gt_460](data/point_eval_f_gt_image_460.png)

Model prediction

![460](data/460_eval.gif)

Ground truth mesh

![point_eval_f_gt_460](data/460_gt_eval.gif)


Sample \#1450

![point_eval_f_gt_1450](data/point_eval_f_gt_image_1450.png)

Model prediction

![1450](data/1450_eval.gif)

Training on one class provides more steady result. The reason might be Training on three classes cause confusion to the model. The chair prediction gets worse. The F1 score across 3 dataset is roughly the same as the F1 socre on chair dataset.

