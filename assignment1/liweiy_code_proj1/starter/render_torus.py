"""
Sample code to render various representations.

Usage:
    python -m starter.render_torus
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
from starter.utils import get_device, get_points_renderer

def render_torus(image_size=256, num_samples=200, device=None):

    if device is None:
        device = get_device()

    u = torch.linspace(0, 2 * np.pi, num_samples)
    v = torch.linspace(0, 2 * np.pi, num_samples)

    U, V = torch.meshgrid(u, v)
    # circle radius, tube radius
    t_R, r = 2, 1.0

    x = (t_R + r * torch.cos(V)) * torch.cos(U)
    y = (t_R + r * torch.cos(V)) * torch.sin(U)
    z = r * torch.sin(V)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    num_views = 36
    R, T = pytorch3d.renderer.look_at_view_transform(6, 0, np.linspace(-180, 180, num_views, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)

    # print(torus_point_cloud.points_packed().shape)

    # fig = plot_scene({
    #     "figure": {
    #         "point_cloud": torus_point_cloud,
    #         "Camera": cameras,
    #     }
    # })
    # fig.show()

    rend = renderer(torus_point_cloud.extend(num_views), cameras=cameras)
    my_images = (rend[:, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return list(my_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    images = render_torus()
    imageio.mimsave('images/torus.gif', images, duration=1000//15, loop=0)

