import argparse
import time
import numpy as np
import imageio

import dataset_location
import losses
import torch
from model import SingleViewto3D
import pytorch3d
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from r2n2_custom import R2N2
import utils_vox
from utils import get_mesh_renderer, get_points_renderer


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=4e-4, type=float)
    # parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--max_iter", default=100000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh"], type=str
    )
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--save_freq", default=2000, type=int)
    # parser.add_argument("--save_freq", default=20, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser


def preprocess(feed_dict, args):
    images = feed_dict["images"].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)


def calculate_loss(predictions, ground_truth, args):
    if args.type == "vox":
        loss = losses.voxel_loss(predictions, ground_truth)
    elif args.type == "point":
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == "mesh":
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth
    return loss


def train_model(args):
    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
        return_feats=args.load_feat,
    )

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    train_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")

    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)

        if args.type == "mesh" and  step % 50 == 0:
            print("visualizing")
            num_views = 36
            R, T = pytorch3d.renderer.look_at_view_transform(
                dist=1.2,
                elev=0,
                azim=np.linspace(-180, 180, num_views, endpoint=False),
            )
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
                R=R,
                T=T,
                device=args.device
            )
            lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=args.device,)
            vertices, faces = prediction_3d.verts_list()[0], prediction_3d.faces_list()[0]
            vertices = vertices.unsqueeze(0)
            faces = faces.unsqueeze(0)
            textures = torch.ones_like(vertices)
            textures = textures * torch.tensor([0.7, 0.7, 1]).to(args.device)
            data = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=pytorch3d.renderer.TexturesVertex(textures)).detach()
            renderer = get_mesh_renderer(image_size=256)
            rend = renderer(data.extend(num_views), cameras=cameras, lights=lights)
            my_images = (rend[:, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            # for i in range(36):
                # plt.imsave(args.output_path[:-4] + f"_{step}_{i}.png", my_images[i])
            # plt.imsave(args.output_path[:-4] + f"_{step}.png", my_images.squeeze())
            imageio.mimsave(args.output_path[:-4] + f"train_{step}.gif", list(my_images), duration=1000//15, loop=0)


        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0 and step > 0:
            print(f"Saving checkpoint at step {step}")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoint_{args.type}.pth",
            )

        print(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
            % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
        )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
