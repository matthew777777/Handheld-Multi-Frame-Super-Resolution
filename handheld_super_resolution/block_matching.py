import time

import numpy as np
from numba import njit, prange
import torch

from .utils import DEFAULT_TORCH_FLOAT_TYPE, round_half_away
from .config import Config

BOX_FILTER_8 = torch.as_tensor(np.ones((1,1,8,8)), dtype=DEFAULT_TORCH_FLOAT_TYPE)
BOX_FILTER_8.requires_grad = False
BOX_FILTER_16 = torch.as_tensor(np.ones((1,1,16,16)), dtype=DEFAULT_TORCH_FLOAT_TYPE)
BOX_FILTER_16.requires_grad = False
BOX_FILTER_32 = torch.as_tensor(np.ones((1,1,32,32)), dtype=DEFAULT_TORCH_FLOAT_TYPE)
BOX_FILTER_32.requires_grad = False
BOX_FILTER_64 = torch.as_tensor(np.ones((1,1,64,64)), dtype=DEFAULT_TORCH_FLOAT_TYPE)
BOX_FILTER_64.requires_grad = False

def align_lvl_block_matching_L2(tyled_pyr_lvl, ref_fft_lvl: torch.Tensor, moving_lvl: torch.Tensor, alignment: torch.Tensor, l: int, config: Config):
    verbose = config.verbose > 2
    currentTime = time.perf_counter()
    tileSize = config.alignment.tile_sizes[l]
    searchRadius = config.alignment.search_radii[l]
    distanceMetric = config.alignment.block_matching.metrics[l]

    imshape = moving_lvl.shape

    search_size = 2 * searchRadius + tileSize # The size of the crop in which the search is done
    corr_size = 2 * searchRadius + 1 # The size of the correlation map output

    # Extract tiles based on the optical flow
    search_area = extract_flow_patches(moving_lvl, alignment, tileSize, searchRadius)

    moving_fft = torch.fft.rfft2(search_area, dim=(-2, -1))
    corrs = torch.fft.irfft2(torch.conj(ref_fft_lvl) * moving_fft, s=(search_size, search_size))
    corrs = torch.fft.fftshift(corrs, dim=(-2, -1))

    # crop to valid region ±R around center output size search_size
    # Before cropping, corrs has an even shaped. The correlation with shift=0 is almost at the center, biased towards the bottom left. By removing 1 pixel from top and left, we center it.
    # The rest of the crop removes the phantom "circular" correlations at the borders due to the FFT
    pre_crop_size = corrs.shape[-1]
    crop = (pre_crop_size - 1 - corr_size) // 2
    corrs = corrs[..., crop+1:crop+corr_size+1, crop+1:crop+corr_size+1]


    ## Now compute the windows L2 norm of the search patches
    if tileSize == 8:
        box_filter = BOX_FILTER_8
    elif tileSize == 16:
        box_filter = BOX_FILTER_16
    elif tileSize == 32:
        box_filter = BOX_FILTER_32
    elif tileSize == 64:
        box_filter = BOX_FILTER_64
    else:
        raise NotImplementedError("Box filter for tile size {} not implemented".format(tileSize))

    # TODO there may be a faster and smarter way than conv2d for this, but this is not the bottleneck so far
    L2_search = torch.nn.functional.conv2d(
        search_area.view(-1, 1, search_size, search_size).square(), box_filter, padding="valid")
    L2_search = L2_search.view(search_area.shape[0], search_area.shape[1], L2_search.shape[-2], L2_search.shape[-1])

    ## Final L2 error computation (Not the full L2, but enough to find the minimum)
    L2_error = L2_search - 2 * corrs

    L2_error_ = L2_error.flatten(-2, -1)
    max_idx = torch.argmin(L2_error_, dim=-1)
    peak_y = max_idx // corr_size
    peak_x = max_idx % corr_size
    dy = peak_y - corr_size//2
    dx = peak_x - corr_size//2

    # Test alignment here
    alignment[:, :, 0] += dx
    alignment[:, :, 1] += dy

def align_lvl_block_matching_L1(ref_lvl: torch.Tensor, moving_lvl: torch.Tensor, alignments: torch.Tensor, l: int, config: Config):
    tile_size = config.alignment.tile_sizes[l]
    search_radius = config.alignment.search_radii[l]

    cpu_l1_local_search(
        ref_lvl.numpy(), moving_lvl.numpy(), search_radius,
        alignments.numpy(), tile_size)


@njit(parallel=True)
def cpu_l1_local_search(ref, moving, search_radius, alignments, tile_size):
    """Exhaustive L1 block matching around the current flow, one tile per task.

    Equivalent to the former per-tile-size CUDA kernels: for each tile, the
    L1 error between the reference tile and the moving image shifted by
    ``flow + (shift_x, shift_y)`` is minimized over the search window, and
    the winning (first minimum in scan order) shift is added to the flow.
    Out-of-bounds moving samples read as 0.
    """
    h, w = moving.shape
    n_tiles_y, n_tiles_x, _ = alignments.shape

    for tile_id in prange(n_tiles_y * n_tiles_x):
        tile_y = tile_id // n_tiles_x
        tile_x = tile_id % n_tiles_x

        flow_x = round_half_away(alignments[tile_y, tile_x, 0])
        flow_y = round_half_away(alignments[tile_y, tile_x, 1])

        best_err = np.float32(np.inf)
        best_shift_x = -search_radius
        best_shift_y = -search_radius
        for shift_y in range(-search_radius, search_radius + 1):
            for shift_x in range(-search_radius, search_radius + 1):
                err = np.float32(0.0)
                for ty in range(tile_size):
                    y = tile_y * tile_size + ty
                    for tx in range(tile_size):
                        x = tile_x * tile_size + tx
                        mov_y = y + flow_y + shift_y
                        mov_x = x + flow_x + shift_x
                        if 0 <= mov_y < h and 0 <= mov_x < w:
                            m = moving[mov_y, mov_x]
                        else:
                            m = np.float32(0.0)
                        diff = ref[y, x] - m
                        err += diff if diff >= 0 else -diff
                if err < best_err:
                    best_err = err
                    best_shift_x = shift_x
                    best_shift_y = shift_y

        alignments[tile_y, tile_x, 0] = flow_x + best_shift_x
        alignments[tile_y, tile_x, 1] = flow_y + best_shift_y


def extract_flow_patches(frame_tgt: torch.Tensor, flow: torch.Tensor, patch_size: int, radius: int):
    ny, nx, _ = flow.shape
    p = patch_size
    r = radius
    P_search = 2 * r + p
    frame_tgt = torch.as_tensor(frame_tgt, dtype=DEFAULT_TORCH_FLOAT_TYPE) # type: ignore
    flow = flow.round().long()

    dx = flow[..., 0]
    dy = flow[..., 1]

    # compute top-left corner of each patch
    top = torch.arange(ny, device=frame_tgt.device)[:, None] * p + dy
    left = torch.arange(nx, device=frame_tgt.device)[None, :] * p + dx

    offsets = torch.arange(P_search, device=frame_tgt.device) - r
    dy_offsets, dx_offsets = torch.meshgrid(offsets, offsets, indexing='ij')

    y_coords = top[:, :, None, None] + dy_offsets[None, None, :, :]
    x_coords = left[:, :, None, None] + dx_offsets[None, None, :, :]

    # clamp to image boundaries
    y_coords = y_coords.clamp(0, frame_tgt.shape[0]-1)
    x_coords = x_coords.clamp(0, frame_tgt.shape[1]-1)

    # flatten for advanced indexing
    y_flat = y_coords.reshape(-1)
    x_flat = x_coords.reshape(-1)

    aligned_patches = frame_tgt[y_flat, x_flat].view((ny, nx, P_search, P_search))
    return aligned_patches
