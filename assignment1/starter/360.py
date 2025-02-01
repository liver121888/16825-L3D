"""
Usage:
    python -m starter.360
"""
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pytorch3d
from pytorch3d.vis.plotly_vis import plot_scene

from starter.utils import get_device, get_mesh_renderer

def full_render(
    cow_path="data/cow.obj",
    image_size=256,
    num_views=360,
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    renderer = get_mesh_renderer(image_size=image_size)

    R, T = pytorch3d.renderer.look_at_view_transform(
        dist=3,
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
    my_images = renderer(meshes.extend(num_views), cameras=cameras, lights=lights)

    fig = plot_scene({
        "figure": {
            "Mesh": meshes,
            "Camera": cameras,
        }
    })
    fig.show()

    # my_images torch.Size([36, 256, 256, 4])
    print("my_images", my_images.shape)
    my_images = my_images[:, ..., :3]
    my_images = my_images.cpu().numpy()
    my_images = (my_images * 255).clip(0, 255).astype(np.uint8)

    image_list = [img for img in my_images]
    return image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/360.gif")
    args = parser.parse_args()
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    image_list = full_render(cow_path=args.cow_path, image_size=args.image_size, num_views=36)
    imageio.mimsave(args.output_path, image_list, duration=duration, loop=0)
