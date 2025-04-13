# Assignment 5

Liwei Yang, liweiy@andrew.cmu.edu

Collaborators: fuchengp, jiamuz, jinkaiq

- [Assignment 5](#assignment-5)
  - [Q1. Classification Model (40 points)](#q1-classification-model-40-points)
    - [Successful predictions](#successful-predictions)
    - [Wrong predictions](#wrong-predictions)
  - [Q2. Segmentation Model (40 points)](#q2-segmentation-model-40-points)
  - [Q3. Robustness Analysis (20 points)](#q3-robustness-analysis-20-points)
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

## Q3. Robustness Analysis (20 points) 

## Q4. Bonus Question - Locality (20 points)