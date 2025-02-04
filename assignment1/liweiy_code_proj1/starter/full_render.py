
import argparse
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pytorch3d
import torch
from pytorch3d.vis.plotly_vis import plot_scene

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh

def full_render(
    cow_path="data/cow.obj",
    image_size=256,
    num_views=36,
    texture = False,
    device=None,
):
    if device is None:
        device = get_device()

    if texture:
        meshes = pytorch3d.io.load_objs_as_meshes([cow_path], device=device)
    else:
        vertices, faces = load_cow_mesh(cow_path)
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor([0.7, 0.7, 1])  # (1, N_v, 3)
        meshes = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        ).to(device)

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
    rend = renderer(meshes.extend(num_views), cameras=cameras, lights=lights)

    # fig = plot_scene({
    #     "figure": {
    #         "Mesh": meshes,
    #         "Camera": cameras,
    #     }
    # })
    # fig.show()

    # my_images torch.Size([36, 256, 256, 4])
    print("rend", rend.shape)
    my_images = (rend[:, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return list(my_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/360.gif")
    args = parser.parse_args()

    image_list = full_render(cow_path=args.cow_path, image_size=args.image_size, num_views=36)
    imageio.mimsave(args.output_path, image_list, duration=1000//15, loop=0)
