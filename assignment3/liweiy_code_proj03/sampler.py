import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_values = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device=ray_bundle.origins.device)

        # TODO (Q1.4): Sample points from z values
        # torch.Size([256**2, 3])
        # torch.Size([64])
        # print(ray_bundle.origins.shape)
        # print(z_vals.shape)

        # torch.Size([256**2, 1, 3])
        # torch.Size([1, 64, 1])

        # broadcasting
        sample_points = ray_bundle.origins[:, None, :] + \
            ray_bundle.directions[:, None, :] * z_values[None, :, None]
        
        sample_lengths = z_values[None, :, None] * torch.ones_like(sample_points[..., :1])
        # print("z_vals: ", torch.min(z_vals), torch.max(z_vals))

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=sample_lengths,
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}