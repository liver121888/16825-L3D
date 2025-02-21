# Assignment 2

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
  - [3. Exploring other architectures / datasets. (Choose at least one! More than one is extra credit)](#3-exploring-other-architectures--datasets-choose-at-least-one-more-than-one-is-extra-credit)

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

### 2.5. Analyse effects of hyperparams variations (10 points)

### 2.6. Interpret your model (15 points)

## 3. Exploring other architectures / datasets. (Choose at least one! More than one is extra credit)
