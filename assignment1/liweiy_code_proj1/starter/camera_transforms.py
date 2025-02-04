"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
from pytorch3d.vis.plotly_vis import plot_scene
from starter.utils import get_device, get_mesh_renderer


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    # R_relative = torch.tensor(R_relative).float()
    # T_relative = torch.tensor(T_relative).float()

    # can push everything into the center then cal the relative R and T

    num_tf = 5

    R_0 = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T_0 = torch.tensor([0.0, 0, 3.0])
    
    # R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    # T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative

    # turning object and turning frame is different
    # if turning object, z 90 actually has tf z -90

    # the numbering here is different from the one in the assignment
    # it follows the numbering in the file name

    # z -90
    # x become y, y become -x
    R1 = torch.tensor([[0.0, 1.0, 0], [-1, 0, 0], [0, 0, 1]])
    T1 = torch.tensor([0.0, 0, 3.0])

    # y 90
    R2 = torch.tensor([[0.0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])
    T2 = torch.tensor([0.0, 0, 3.0])

    R3 = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T3 = torch.tensor([0.0, 0, 5.0])

    R4 = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T4 = torch.tensor([0.5, -0.5, 3.0])

    Rs = [R_0, R1, R2, R3, R4]
    Ts = [T_0, T1, T2, T3, T4]
    R = []
    T = []
    for i in range(num_tf):
        R.append(Rs[i])
        T.append(Ts[i])

    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.stack(Rs), T=torch.stack(Ts), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes.extend(num_tf), cameras=cameras, lights=lights)

    # fig = plot_scene({
    #     "figure": {
    #         "Mesh": meshes,
    #         "Camera": cameras,
    #     }
    # })
    # fig.show()

    # TODO: print out relative R and T
    # R = R_relative @ R_0
    # T = R_relative @ T_0 + T_relative
    # R_relative1 = R1 @ R_0.inverse()
    # T_relative1 = T1 - R_relative1 @ T_0

    R_relative1 = R1 @ R_0.inverse()
    T_relative1 = T1 - R_relative1 @ T_0

    R_relative2 = R2 @ R_0.inverse()
    T_relative2 = T2 - R_relative2 @ T_0

    R_relative3 = R3 @ R_0.inverse()
    T_relative3 = T3 - R_relative3 @ T_0

    R_relative4 = R4 @ R_0.inverse()
    T_relative4 = T4 - R_relative4 @ T_0

    print("R_relative1", R_relative1)
    print("T_relative1", T_relative1)
    print("R_relative2", R_relative2)
    print("T_relative2", T_relative2)
    print("R_relative3", R_relative3)
    print("T_relative3", T_relative3)
    print("R_relative4", R_relative4)
    print("T_relative4", T_relative4)
    
    # print(rend.shape)
    return rend[:, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/textured_cow.jpg")
    args = parser.parse_args()
    images = render_textured_cow(cow_path=args.cow_path, image_size=args.image_size)
    for i, image in enumerate(images):
        plt.imsave(f"images/tfed_cam{i}.jpg", image)
