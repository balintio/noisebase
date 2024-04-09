"""
noisebase.torch.projective
--------------------------

also available under noisebase.torch
"""

import torch
from noisebase.torch import tensor_like

def backproject_pixel_centers(motion, crop_offset, prev_crop_offset, as_grid = False):
    """Decompresses per-sample radiance from RGBE compressed data

    Args:
        motion (tensor, N2HW): Per-sample screen-space motion vectors (in pixels) 
            see `noisebase.projective.motion_vectors`
        crop_offset (tensor, size (2)): offset of random crop (window) from top left corner of camera frame (in pixels)
        prev_crop_offset (tensor, size (2)): offset of random crop (window) in previous frame
        as_grid (bool): torch.grid_sample, with align_corners = False format

    Returns:
        pixel_position (tensor, N2HW): ij indexed pixel coordinates OR
        pixel_position (tensor, NHW2): xy WH position (-1, 1) IF as_grid
    """
    height = motion.shape[2]
    width = motion.shape[3]
    dtype = motion.dtype
    device = motion.device

    pixel_grid = torch.stack(torch.meshgrid(
        torch.arange(0, height, dtype=dtype, device=device),
        torch.arange(0, width, dtype=dtype, device=device),
        indexing='ij'
    ))

    pixel_pos = pixel_grid + motion - prev_crop_offset[..., None, None] + crop_offset[..., None, None]

    if as_grid:
        # as needed for grid_sample, with align_corners = False
        pixel_pos_xy = torch.permute(torch.flip(pixel_pos, (1,)), (0, 2, 3, 1)) + 0.5
        image_pos = pixel_pos_xy / tensor_like(pixel_pos, [width, height])
        return image_pos * 2 - 1
    else:
        return pixel_pos