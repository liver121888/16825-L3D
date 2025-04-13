import numpy as np
import argparse
import os
import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg
from pytorch3d.transforms import Rotate

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='/mnt/data/assignment5/data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='/mnt/data/assignment5/data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--rotation', type=float, default=None, help='rotation angle in degrees')
    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device)
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind]).to(args.device)

    print("test data shape: {}".format(test_data.shape))
    # test data shape: torch.Size([617, 10000, 3])
    print("test label shape: {}".format(test_label.shape))
    # test label shape: torch.Size([617, 10000])

    if args.rotation is not None:
        rotation = args.rotation / 180 * np.pi
        R0 = torch.tensor([[1., 0., 0.],
                            [0., float(np.cos(rotation)), float(np.sin(rotation))],
                            [0., float(-np.sin(rotation)), float(np.cos(rotation))]]).unsqueeze(0)
        R0 = torch.tile(R0, (test_data.shape[0], 1, 1)).to(args.device)
        trans = Rotate(R0)
        test_data = trans.transform_points(test_data)

    # ------ TO DO: Make Prediction ------
    data_loader = torch.split(test_data, 32)
    label_loader = torch.split(test_label, 32)
    pred_labels = []

    if os.path.exists(args.output_dir + "/seg"):
        pass
    else:
        os.makedirs(args.output_dir + "/seg")

    # 5 objects, 2 bad, 3 good class
    bad_cnt = 0
    good_cnt = 0
    for data, label in zip(data_loader, label_loader):

        pred_label = torch.argmax(model(data), dim=2)
        pred_labels.append(pred_label)
        # print("data shape: {}".format(data.shape))
        # print("label shape: {}".format(label.shape))
        # print("pred label shape: {}".format(pred_label.shape))

        for i in range(label.shape[0]):
            pcl = data[i]
            # ([10000, 3])
            # print("pcl shape: {}".format(pcl.shape))
            l = label[i]
            p_l = pred_label[i]

            equal_mask = p_l == l
            accuracy = equal_mask.sum().item() / equal_mask.numel()

            if accuracy < 0.5:
                if bad_cnt < 2:
                    print("bad prediction {}: accuracy {}".format(bad_cnt, accuracy))
                    viz_seg(pcl, l, args.output_dir + "/seg/bad_gt_{}.gif".format(bad_cnt), args.device, args.num_points)
                    viz_seg(pcl, p_l, args.output_dir + "/seg/bad_pred_{}.gif".format(bad_cnt), args.device, args.num_points)
                    bad_cnt += 1
            elif accuracy > 0.90:
                if good_cnt < 3:
                    print("good prediction {}: accuracy {}".format(good_cnt, accuracy))
                    viz_seg(pcl, l, args.output_dir + "/seg/good_gt_{}.gif".format(good_cnt), args.device, args.num_points)
                    viz_seg(pcl, p_l, args.output_dir + "/seg/good_pred_{}.gif".format(good_cnt), args.device, args.num_points)
                    good_cnt += 1

    # print(test_label[args.i])

    pred_labels = torch.cat(pred_labels)
    print("pred labels shape: {}".format(pred_labels.shape))
    print("test label shape: {}".format(test_label.shape))

    test_accuracy = pred_labels.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Visualize Segmentation Result (Pred VS Ground Truth)
    # viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}_{}.gif".format(args.output_dir + "/seg", args.exp_name, args.i), args.device)
    # viz_seg(test_data[args.i], pred_labels[args.i], "{}/pred_{}_{}.gif".format(args.output_dir + "/seg", args.exp_name,args.i), args.device)
