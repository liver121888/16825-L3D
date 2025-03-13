#!/bin/bash

# python volume_rendering_main.py --config-name=box

# python volume_rendering_main.py --config-name=train_box

# python volume_rendering_main.py --config-name=nerf_lego

# set back ratio!!
# view dependent false
# python volume_rendering_main.py --config-name=nerf_materials_highres

# set view dependent true
python volume_rendering_main.py --config-name=nerf_materials_highres_view_dependent

python volume_rendering_main.py --config-name=nerf_lego_highres

python volume_rendering_main.py --config-name=nerf_lego_highres_view_dependent

# python -m surface_rendering_main --config-name=torus_surface

# python -m surface_rendering_main --config-name=points_surface

# python -m surface_rendering_main --config-name=volsdf_surface