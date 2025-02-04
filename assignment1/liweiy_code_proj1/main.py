import imageio
import matplotlib.pyplot as plt
import torch
import os

from starter.utils import unproject_depth_image
from starter.full_render import full_render
from starter.dolly_zoom import dolly_zoom
from starter.render_tetrahedron import render_tetrahedron
from starter.render_cube import render_cube
from starter.retexturing_cow import render_cow
from starter.camera_transforms import render_textured_cow
from starter.render_pointcloud import load_rgbd_data, render_plant
from starter.render_torus import render_torus
from starter.render_klein_bottle import render_kb
from starter.render_torus_mesh import render_torus_mesh
from starter.render_saddle_mesh import render_saddle_mesh
from starter.retexturing_rpg import render_textured_rpg
from starter.sampling_cow import sample_cow

def create_gif(images, filename="output.gif"):
    imageio.mimsave(filename, images, duration=1000//15, loop=0)

def main():

    output_dir = 'output/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            os.remove(output_dir + file)

    # 360
    images = full_render(texture = False)
    create_gif(images,  output_dir + '360_nt.gif')
    images = full_render(texture = True)
    create_gif(images,  output_dir + '360.gif')

    # dolly_zoom
    images = dolly_zoom()
    create_gif(images, output_dir + 'dolly.gif')

    # tetrahedron
    images = render_tetrahedron()
    create_gif(images, output_dir + 'tetrahedron_render.gif')

    # cube
    images = render_cube()
    create_gif(images, output_dir + 'cube_render.gif')

    # retexturing
    images = render_cow()
    create_gif(images, output_dir + 'cow_retextured.gif')

    # camera tf
    images = render_textured_cow()
    for i, image in enumerate(images):
        plt.imsave(output_dir + f"tfed_cam{i}.jpg", image)

    # plant
    rgb_data = load_rgbd_data()
    print(rgb_data.keys())
    p1, c1 = unproject_depth_image(torch.tensor(rgb_data['rgb1']), 
                                   torch.tensor(rgb_data['mask1']), 
                                   torch.tensor(rgb_data['depth1']), 
                                   rgb_data['cameras1'])
    
    p2, c2 = unproject_depth_image(torch.tensor(rgb_data['rgb2']), 
                                   torch.tensor(rgb_data['mask2']), 
                                   torch.tensor(rgb_data['depth2']), 
                                   rgb_data['cameras2'])
    # flip y and x
    p1[:, :2] *= -1
    p2[:, :2] *= -1
    images = render_plant(p1, c1)
    create_gif(images, output_dir + 'plant1.gif')
    images = render_plant(p2, c2)
    create_gif(images, output_dir + 'plant2.gif')
    print(p1.shape, p2.shape)
    p_all = torch.cat((p1, p2), dim=0)
    c_all = torch.cat((c1, c2), dim=0)
    images = render_plant(p_all, c_all)
    create_gif(images, output_dir + 'plantall.gif')

    # torus
    images = render_torus()
    create_gif(images, output_dir + 'torus.gif')

    # klein bottle
    images = render_kb()
    create_gif(images, output_dir + 'klein.gif')

    # torus mesh
    images = render_torus_mesh()
    create_gif(images, output_dir + 'torus_mesh.gif')

    # saddle mesh
    images = render_saddle_mesh()
    create_gif(images, output_dir + 'saddle_mesh.gif')

    # rpg
    images = render_textured_rpg()
    create_gif(images, output_dir + 'textured_rpg.gif')

    # sampling
    num_samples = [10, 100, 1000, 10000]
    for ns in num_samples:
        images = sample_cow(num_samples=ns)
        create_gif(images, output_dir + f'cow_sampled_{ns}.gif')

if __name__ == "__main__":
    main()