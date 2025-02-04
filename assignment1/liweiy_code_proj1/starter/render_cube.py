"""
Usage:
    python -m starter.render_cube --image_size 256
"""
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer

def render_cube(image_size=256, color=[0.7, 0.7, 1], num_views=36, device=None):

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
    vertices = torch.tensor(
        [
            [1, 1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, -1],
            [1, -1, -1],
        ],
        dtype=torch.float32,
    )
    
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
            [4, 7, 6],
            [4, 6, 5],
        ]
    )

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=4.0,
        elev=0,
        azim=np.linspace(-180, 180, num_views, endpoint=False),
    )
    print("R", R.shape)
    print("T", T.shape)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        device=device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    my_images = renderer(mesh.extend(num_views), cameras=cameras, lights=lights)

    # my_images torch.Size([36, 256, 256, 4])
    print("my_images", my_images.shape)
    my_images = my_images[:, ..., :3]
    my_images = my_images.cpu().numpy()
    my_images = (my_images * 255).clip(0, 255).astype(np.uint8)

    image_list = [img for img in my_images]
    return image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="images/cube_render.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    image_list = render_cube(image_size=args.image_size)
    # plt.imsave(args.output_path, image)
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave(args.output_path, image_list, duration=duration, loop=0)
