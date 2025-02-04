"""
Usage:
    python -m starter.retexturing_rpg --image_size 512
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pytorch3d
import imageio
import torch
from pytorch3d.transforms import Rotate
from pytorch3d.vis.plotly_vis import plot_scene
from starter.utils import get_device, get_mesh_renderer


def render_textured_rpg(
    rpg_path="data/rpg.obj",
    image_size=256,
    device=None,
):
    if device is None:
        device = get_device()

    vertices, face_props, text_props = pytorch3d.io.load_obj(rpg_path)
    faces = face_props.verts_idx
    # vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    verts_uvs = text_props.verts_uvs
    faces_uvs = face_props.textures_idx

    texture_map = plt.imread("data/rpg.png")
    texture_map = texture_map[..., :3]

    textures_uv = pytorch3d.renderer.TexturesUV(
        maps=torch.tensor(np.array([texture_map])),
        faces_uvs=faces_uvs.unsqueeze(0),
        verts_uvs=verts_uvs.unsqueeze(0),
    ).to(device)

    relative_rotation = pytorch3d.transforms.euler_angles_to_matrix(
        torch.tensor([-np.pi/2, 0, 0]), "XYZ"
    )
    print("relative_rotation", relative_rotation)
    print("vertices", vertices.shape)

    # Apply transformation
    meshes = pytorch3d.structures.Meshes(
        verts=(vertices @ relative_rotation.T).unsqueeze(0),
        faces=faces,
        textures=textures_uv,
    ).to(device)

    renderer = get_mesh_renderer(image_size=image_size)
    num_views = 36
    R, T = pytorch3d.renderer.look_at_view_transform(2, 0, azim=np.linspace(-180, 180, num_views, endpoint=False))
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R, T=T, device=device,
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
    
    print("rend", rend.shape)
    my_images = (rend[:, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return list(my_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpg_path", type=str, default="data/rpg.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/textured_rpg.gif")
    args = parser.parse_args()
    images = render_textured_rpg(rpg_path=args.rpg_path, image_size=args.image_size)
    imageio.mimsave(args.output_path, images, duration=1000//15, loop=0)

