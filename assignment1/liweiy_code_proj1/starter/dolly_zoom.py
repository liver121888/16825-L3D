"""
Usage:
    python -m starter.dolly_zoom --num_frames 10
"""

import argparse

import imageio
import numpy as np
import pytorch3d
from pytorch3d.vis.plotly_vis import plot_scene
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from starter.utils import get_device, get_mesh_renderer


def dolly_zoom(
    image_size=256,
    num_frames=10,
    device=None,
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(5, 120, num_frames)
    print(fovs)

    Ts = []
    width = 5
    for fov in tqdm(fovs):
        distance = width/(2 * np.tan(np.deg2rad(fov/2)))
        # print(distance)
        T = [0, 0, distance]
        Ts.append(T)
    Ts = torch.tensor(Ts)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fovs, T=Ts, device=device)
    rend = renderer(mesh.extend(num_frames), cameras=cameras, lights=lights)

    rend = rend[:, ..., :3]
    rend = rend.cpu().numpy()

    images = []
    for i, r in enumerate(rend):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))

    # fig = plot_scene({
    #     "figure": {
    #         "Mesh": mesh,
    #         "Camera": cameras,
    #     }
    # })

    # fig.show()
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output_file", type=str, default="images/dolly.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    dolly_zoom(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )
