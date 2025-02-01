"""
Sample code to render a cow.

Usage:
    python -m starter.retexturing_cow --image_size 256 --output_path images/cow_retextured.jpg
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pytorch3d
import torch
from pytorch3d.vis.plotly_vis import plot_scene

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow(
    cow_path="data/cow.obj", image_size=256, color1=np.array([1.0, 0.0, 1]), color2=np.array([1.0, 1.0, 0.0]), device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)

    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    # textures = textures * torch.tensor(color)  # (1, N_v, 3)

    z_min = vertices[:, :, 2].min()
    z_max = vertices[:, :, 2].max()
    for i in range(vertices.shape[1]):
        z = vertices[0, i, 2]
        alpha = (z - z_min) / (z_max - z_min)
        x = torch.tensor(alpha.item() * color2 + (1 - alpha.item()) * color1)
        # print(x)
        textures[0, i, :] = x

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    R = torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).unsqueeze(0)


    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R, T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    # fig = plot_scene({
    #     "figure": {
    #         "Mesh": mesh,
    #         "Camera": cameras,
    #     }
    # })
    # fig.show()


    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_retextured.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    plt.imsave(args.output_path, image)
