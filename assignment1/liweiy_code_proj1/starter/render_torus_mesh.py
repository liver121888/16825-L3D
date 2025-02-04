"""
Sample code to render various representations.

Usage:
    python -m starter.render_torus_mesh
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
from starter.utils import get_device, get_mesh_renderer

def render_torus_mesh(image_size=256, voxel_size=64, device=None):

    if device is None:
        device = get_device()
    min_value = -3.5
    max_value = 3.5
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

    # circle radius, tube radius
    t_R, r = 2, 1.0
    voxels = (torch.sqrt(X ** 2 + Y ** 2) - t_R)**2 + Z ** 2 - r ** 2

    # voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device, lights=lights)

    num_views = 36
    R, T = pytorch3d.renderer.look_at_view_transform(6, 0, np.linspace(-180, 180, num_views, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    # fig = plot_scene({
    #     "figure": {
    #         "mesh": mesh,
    #         "Camera": cameras,
    #     }
    # })
    # fig.show()

    rend = renderer(mesh.extend(num_views), cameras=cameras)
    my_images = (rend[:, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return list(my_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    images = render_torus_mesh()
    imageio.mimsave('images/torus_mesh.gif', images, duration=1000//15, loop=0)

