"""
Sample code to render various representations.

Usage:
    python -m starter.render_pointcloud
"""
import argparse
import pickle
import imageio
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
from pytorch3d.vis.plotly_vis import plot_scene
from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_plant(p, c, device = None):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()

    v1 = p.to(device).unsqueeze(0)
    r1 = c.to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=v1, features=r1)

    num_views = 36
    R, T = pytorch3d.renderer.look_at_view_transform(6, 0, np.linspace(-180, 180, num_views, endpoint=False))
    # print("R", R.shape)
    # print("T", T.shape)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )

    # fig = plot_scene({
    #     "figure": {
    #         "point_cloud": point_cloud,
    #         "Camera": cameras,
    #     }
    # })
    # fig.show()

    renderer = get_points_renderer(
        image_size=256, background_color=(1, 1, 1)
    )

    rend = renderer(point_cloud.extend(num_views), cameras=cameras)
    my_images = (rend[:, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return list(my_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()


    rgb_data = load_rgbd_data()
    print(rgb_data.keys())

    p1, c1 = unproject_depth_image(torch.tensor(rgb_data['rgb1']), 
                                   torch.tensor(rgb_data['mask1']), 
                                   torch.tensor(rgb_data['depth1']), 
                                   rgb_data['cameras1'])
    
    p2, c2 = unproject_depth_image(torch.tensor(rgb_data['rgb2']), 
                                   torch.tensor(rgb_data['mask2']), 
                                   torch.tensor(rgb_data['depth2']), 
                                   rgb_data['cameras2'])
    
    # flip y and x
    p1[:, :2] *= -1
    p2[:, :2] *= -1


    images = render_plant(p1, c1)
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave('images/plant1.gif', images, duration=duration, loop=0)

    images = render_plant(p2, c2)
    imageio.mimsave('images/plant2.gif', images, duration=duration, loop=0)

    print(p1.shape, p2.shape)

    p_all = torch.cat((p1, p2), dim=0)
    c_all = torch.cat((c1, c2), dim=0)
    images = render_plant(p_all, c_all)
    imageio.mimsave('images/plantall.gif', images, duration=duration, loop=0)


