# Assignment 5

Liwei Yang, liweiy@andrew.cmu.edu

Collaborators: fuchengp, jiamuz, jinkaiq

- [Assignment 5](#assignment-5)
  - [Q1. Classification Model (40 points)](#q1-classification-model-40-points)
    - [Successful predictions](#successful-predictions)
    - [Wrong predictions](#wrong-predictions)
  - [Q2. Segmentation Model (40 points)](#q2-segmentation-model-40-points)
    - [Good predictions](#good-predictions)
    - [Bad predictions](#bad-predictions)
  - [Q3. Robustness Analysis (20 points)](#q3-robustness-analysis-20-points)
    - [Classification Model](#classification-model)
      - [Rotation](#rotation)
        - [Successful predictions](#successful-predictions-1)
        - [Wrong predictions](#wrong-predictions-1)
      - [Less points](#less-points)
        - [Successful predictions](#successful-predictions-2)
        - [Wrong predictions](#wrong-predictions-2)
    - [Segmentation Model](#segmentation-model)
      - [Rotation](#rotation-1)
        - [Good predictions](#good-predictions-1)
        - [Bad predictions](#bad-predictions-1)
      - [Less points](#less-points-1)
        - [Good predictions](#good-predictions-2)
        - [Bad predictions](#bad-predictions-2)
  - [Q4. Bonus Question - Locality (20 points)](#q4-bonus-question---locality-20-points)

## Q1. Classification Model (40 points)

I trained the model for 120 epochs, the test accuraccy is 97.7%.

### Successful predictions
|Sample|Sample 1|Sample 2|Sample 3|
|:--:|:--:|:--:|:--:|
|Label|Class 0 (Chair)|Class 1 (Vase)|Class 2 (Lamp)|
|Point Cloud|![good_pred_0](data/cls/good_pred_0.gif)|![good_pred_1](data/cls/good_pred_1.gif)|![good_pred_2](data/cls/good_pred_2.gif)|

### Wrong predictions
|Sample|Sample 1|Sample 2|
|:--:|:--:|:--:|
|Gt Label|Class 0 (Chair)|Class 1 (Vase)|
|Pred Label|Class 2 (Lamp)|Class 2 (Lamp)|
|Point Cloud|![bad_pred_0](data/cls/bad_pred_0.gif)|![bad_pred_1](data/cls/bad_pred_1.gif)|

TODO

## Q2. Segmentation Model (40 points) 

I trained the model for 40 epochs, the test accuraccy is 89.5%.

### Good predictions
|Sample|Sample 1|Sample 2|Sample 3|
|:--:|:--:|:--:|:--:|
|Prediction|![good_pred_0](data/seg/good_pred_0.gif)|![good_pred_1](data/seg/good_pred_1.gif)|![good_pred_2](data/seg/good_pred_2.gif)|
|Ground Truth|![good_gt_0](data/seg/good_gt_0.gif)|![good_gt_1](data/seg/good_gt_1.gif)|![good_gt_2](data/seg/good_gt_2.gif)|
|Accuracy|98.5%|96.6%|97.7%|

### Bad predictions
|Sample|Sample 1|Sample 2|
|:--:|:--:|:--:|
|Prediction|![bad_pred_0](data/seg/bad_pred_0.gif)|![bad_pred_1](data/seg/bad_pred_1.gif)|
|Ground Truth|![bad_gt_0](data/seg/bad_gt_0.gif)|![bad_gt_1](data/seg/bad_gt_1.gif)|
|Accuracy|38.2%|40.4%|

## Q3. Robustness Analysis (20 points) 

### Classification Model

#### Rotation
30 degrees rotation

test accuracy: 82.6%

##### Successful predictions
|Sample|Sample 1|Sample 2|Sample 3|
|:--:|:--:|:--:|:--:|
|Label|Class 0 (Chair)|Class 1 (Vase)|Class 2 (Lamp)|
|Point Cloud|![good_pred_0_rotation](data/cls_rotation/good_pred_0.gif)|![good_pred_1_rotation](data/cls_rotation/good_pred_1.gif)|![good_pred_2_rotation](data/cls_rotation/good_pred_2.gif)|

##### Wrong predictions
|Sample|Sample 1|Sample 2|
|:--:|:--:|:--:|
|Gt Label|Class 0 (Chair)|Class 0 (Chair)|
|Pred Label|Class 2 (Lamp)|Class 2 (Lamp)|
|Point Cloud|![bad_pred_0_rotation](data/cls_rotation/bad_pred_0.gif)|![bad_pred_1_rotation](data/cls_rotation/bad_pred_1.gif)|

#### Less points

2000 points

test accuracy: 0.9758656873032528

##### Successful predictions
|Sample|Sample 1|Sample 2|Sample 3|
|:--:|:--:|:--:|:--:|
|Label|Class 0 (Chair)|Class 1 (Vase)|Class 2 (Lamp)|
|Point Cloud|![good_pred_0_points](data/cls_points/good_pred_0.gif)|![good_pred_1_points](data/cls_points/good_pred_1.gif)|![good_pred_2_points](data/cls_points/good_pred_2.gif)|

##### Wrong predictions
|Sample|Sample 1|Sample 2|
|:--:|:--:|:--:|
|Gt Label|Class 0 (Chair)|Class 1 (Vase)|
|Pred Label|Class 2 (Lamp)|Class 2 (Lamp)|
|Point Cloud|![bad_pred_0_points](data/cls_points/bad_pred_0.gif)|![bad_pred_1_points](data/cls_points/bad_pred_1.gif)|

### Segmentation Model

#### Rotation

30 degrees rotation

test accuracy: 75.8%

##### Good predictions
|Sample|Sample 1|Sample 2|Sample 3|
|:--:|:--:|:--:|:--:|
|Prediction|![good_pred_0_rotation](data/seg_rotation/good_pred_0.gif)|![good_pred_1_rotation](data/seg_rotation/good_pred_1.gif)|![good_pred_2_rotation](data/seg_rotation/good_pred_2.gif)|
|Ground Truth|![good_gt_0_rotation](data/seg_rotation/good_gt_0.gif)|![good_gt_1_rotation](data/seg_rotation/good_gt_1.gif)|![good_gt_2_rotation](data/seg_rotation/good_gt_2.gif)|
|Accuracy|90.7%|90.8%|92.0%|

##### Bad predictions
|Sample|Sample 1|Sample 2|
|:--:|:--:|:--:|
|Prediction|![bad_pred_0_rotation](data/seg_rotation/bad_pred_0.gif)|![bad_pred_1_rotation](data/seg_rotation/bad_pred_1.gif)|
|Ground Truth|![bad_gt_0_rotation](data/seg_rotation/bad_gt_0.gif)|![bad_gt_1_rotation](data/seg_rotation/bad_gt_1.gif)|
|Accuracy|42.0%|29.3%|

#### Less points

2000 points

test accuracy: 0.8959611021069692

##### Good predictions
|Sample|Sample 1|Sample 2|Sample 3|
|:--:|:--:|:--:|:--:|
|Prediction|![good_pred_0_points](data/seg_points/good_pred_0.gif)|![good_pred_1_points](data/seg_points/good_pred_1.gif)|![good_pred_2_points](data/seg_points/good_pred_2.gif)|
|Ground Truth|![good_gt_0_points](data/seg_points/good_gt_0.gif)|![good_gt_1_points](data/seg_points/good_gt_1.gif)|![good_gt_2_points](data/seg_points/good_gt_2.gif)|
|Accuracy|95.1%|98.8%|90.5%|

##### Bad predictions
|Sample|Sample 1|Sample 2|
|:--:|:--:|:--:|
|Prediction|![bad_pred_0_points](data/seg_points/bad_pred_0.gif)|![bad_pred_1_points](data/seg_points/bad_pred_1.gif)|
|Ground Truth|![bad_gt_0_points](data/seg_points/bad_gt_0.gif)|![bad_gt_1_points](data/seg_points/bad_gt_1.gif)|
|Accuracy|36.0%|40.2%|


## Q4. Bonus Question - Locality (20 points)