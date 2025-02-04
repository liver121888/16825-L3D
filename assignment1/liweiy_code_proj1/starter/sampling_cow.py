"""
Sample code to render a cow.

Usage:
    python -m starter.sampling_cow --image_size 256
"""
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pytorch3d
import torch
from pytorch3d.vis.plotly_vis import plot_scene

from starter.utils import get_device, get_points_renderer, load_cow_mesh


def sample_cow(
    cow_path="data/cow.obj", image_size=256, num_samples=10, device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_points_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    total_area = 0
    probabilities = []
    for face in faces:
        a = vertices[face[1]] - vertices[face[0]] 
        b = vertices[face[2]] - vertices[face[0]]
        area = 0.5 * np.linalg.norm(np.cross(a, b))
        probabilities.append(area)
        total_area += area

    # print("total_area", total_area)
    # print(sum(probabilities))
    probabilities = np.array(probabilities) / total_area

    points = []
    for _ in range(num_samples):
        sampled_idx = np.random.choice(len(probabilities), p=probabilities)
        face = faces[sampled_idx]
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        r1, r2 = np.random.rand(), np.random.rand()
        u = np.sqrt(r1)
        v = r2 * u
        w = 1 - u
        sampled_point = w * v0 + u * v1 + v * v2
        points.append(sampled_point)

    # print("points", points)
    points = torch.stack(points)
    c = torch.tensor([[0.5, 0.5, 1.0] for _ in range(num_samples)], dtype=torch.float32)

    v = points.to(device).unsqueeze(0)
    r = c.to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=v, features=r)

    num_views = 36
    R, T = pytorch3d.renderer.look_at_view_transform(6, 0, np.linspace(-180, 180, num_views, endpoint=False))

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

    rend = renderer(point_cloud.extend(num_views), cameras=cameras)
    my_images = (rend[:, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return list(my_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    # 10, 100, 1000, 10000
    num_samples = 10000
    images = sample_cow(cow_path=args.cow_path, image_size=args.image_size, num_samples=num_samples)
    # plt.imsave(args.output_path, image)
    imageio.mimsave(f'images/cow_sampled_{num_samples}.gif', images, duration=1000//15, loop=0)
