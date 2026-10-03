import math
import warnings

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
import torch
import torch.nn.functional as F

from .utils import clamp, DEFAULT_NUMPY_FLOAT_TYPE, DEFAULT_TORCH_FLOAT_TYPE
from .config import Config

SOBEL_Y = torch.as_tensor(np.array([[-1], [0], [1]]), dtype=DEFAULT_TORCH_FLOAT_TYPE)[None, None]
SOBEL_X = torch.as_tensor(np.array([[-1,0,1]]), dtype=DEFAULT_TORCH_FLOAT_TYPE)[None, None]
SOBEL_Y.requires_grad = False
SOBEL_X.requires_grad = False

def init_ica(image: torch.Tensor, tile_size: int, config: Config):
    imsize_y, imsize_x = image.shape
    n_patch_y = imsize_y // tile_size
    n_patch_x = imsize_x // tile_size

    gradx = F.conv2d(image[None, None], SOBEL_X, padding='same').squeeze()
    grady = F.conv2d(image[None, None], SOBEL_Y, padding='same').squeeze()

    gradx = gradx.numpy()
    grady = grady.numpy()

    hessian = np.empty((n_patch_y, n_patch_x, 2, 2), DEFAULT_NUMPY_FLOAT_TYPE)

    cpu_compute_hessian(gradx, grady, tile_size, hessian)

    return gradx, grady, hessian

@njit(parallel=True)
def cpu_compute_hessian(gradx, grady, tile_size, hessian):
    n_patchy, n_patch_x, _, _ = hessian.shape

    for patch_id in prange(n_patchy * n_patch_x):
        patch_idy = patch_id // n_patch_x
        patch_idx = patch_id % n_patch_x

        patch_pos_idx = tile_size * patch_idx # global position on the coarse grey grid. Because of extremity padding, it can be out of bound
        patch_pos_idy = tile_size * patch_idy

        h00 = np.float32(0)
        h01 = np.float32(0)
        h11 = np.float32(0)

        for i in range(tile_size):
            for j in range(tile_size):
                pixel_global_idy = patch_pos_idy + i
                pixel_global_idx = patch_pos_idx + j

                local_gradx = gradx[pixel_global_idy, pixel_global_idx]
                local_grady = grady[pixel_global_idy, pixel_global_idx]

                h00 += local_gradx*local_gradx
                h01 += local_gradx*local_grady
                h11 += local_grady*local_grady

        hessian[patch_idy, patch_idx, 0, 0] = h00
        hessian[patch_idy, patch_idx, 0, 1] = h01
        hessian[patch_idy, patch_idx, 1, 0] = h01
        hessian[patch_idy, patch_idx, 1, 1] = h11

def align_lvl_ica(ref_img: torch.Tensor,
                  ref_gradx_lvl: NDArray,
                  ref_grady_lvl: NDArray,
                  ref_hessian_lvl: NDArray,
                  moving_lvl: torch.Tensor,
                  alignment: torch.Tensor,
                  l: int, config: Config):
    verbose_3 = config.verbose >= 3
    tile_size = config.alignment.tile_sizes[l]
    if config.alignment.ica.clip:
        search_radius = config.alignment.search_radii[l]
    else:
        search_radius = 32_767 # Max int16 possible

    if tile_size == 64:
        warnings.warn("Required radius clipping for ica with patchsize 64: Not supported, clipping will be ignored")
        search_radius = 32_767

    cpu_ica(ref_img.numpy(), ref_gradx_lvl, ref_grady_lvl, ref_hessian_lvl,
            moving_lvl.numpy(), alignment.numpy(),
            config.alignment.ica.n_iter, search_radius, tile_size)

@njit(parallel=True)
def cpu_ica(ref_img, gradx, grady, hessian, moving, alignment, niter, clip_radius, tile_size):
    """Inverse-compositional alignment, one tile per task.

    Equivalent to the former per-tile-size CUDA kernels. Each tile's flow is
    refined with ``niter`` Gauss-Newton steps using the precomputed Hessian.
    Preserved per-tile-size semantics: 8x8 tiles sample with clamped borders
    and clip partial sums during reduction (no update clipping), while larger
    tiles sample zero outside the image and clip the update step.
    """
    h, w = moving.shape
    n_tiles_y, n_tiles_x, _ = alignment.shape

    for tile_id in prange(n_tiles_y * n_tiles_x):
        tile_y = tile_id // n_tiles_x
        tile_x = tile_id % n_tiles_x

        A00 = hessian[tile_y, tile_x, 0, 0]
        A01 = hessian[tile_y, tile_x, 0, 1]
        A10 = hessian[tile_y, tile_x, 1, 0]
        A11 = hessian[tile_y, tile_x, 1, 1]

        det = A00 * A11 - A01 * A10
        if abs(det) < np.float32(1e-10): # system is not solvable
            continue
        det_inv = np.float32(1.0) / det

        ax = alignment[tile_y, tile_x, 0]
        ay = alignment[tile_y, tile_x, 1]

        # 8x8 accumulation buffers (tree reduction with clipping, as before)
        c0 = np.empty(64, np.float32)
        c1 = np.empty(64, np.float32)

        for _ in range(niter):
            # Warp I with W(x; p) to compute I(W(x; p))
            if tile_size == 8:
                idx = 0
                for ty in range(8):
                    y = tile_y * 8 + ty
                    for tx in range(8):
                        x = tile_x * 8 + tx
                        ## bilinear interpolation at (x + ax, y + ay), clamped borders
                        floor_x = math.floor(x + ax)
                        floor_y = math.floor(y + ay)
                        frac_x = x + ax - floor_x
                        frac_y = y + ay - floor_y

                        floor_x = clamp(floor_x, 0, w - 1)
                        floor_y = clamp(floor_y, 0, h - 1)

                        ceil_x = clamp(floor_x + 1, 0, w - 1)
                        ceil_y = clamp(floor_y + 1, 0, h - 1)

                        m00 = moving[floor_y, floor_x]
                        m01 = moving[floor_y, ceil_x]
                        m10 = moving[ceil_y, floor_x]
                        m11 = moving[ceil_y, ceil_x]

                        lerpx_top = m00 + (m01 - m00) * frac_x
                        lerpx_bot = m10 + (m11 - m10) * frac_x
                        mov_interp = lerpx_top + (lerpx_bot - lerpx_top) * frac_y

                        gradt = mov_interp - ref_img[y, x]
                        c0[idx] = -gradx[y, x] * gradt
                        c1[idx] = -grady[y, x] * gradt
                        idx += 1
                # tree reduction, clipping the added operand (as before)
                n = 32
                while n > 0:
                    for tid in range(n):
                        c0[tid] += min(max(c0[tid + n], -clip_radius), clip_radius)
                        c1[tid] += min(max(c1[tid + n], -clip_radius), clip_radius)
                    n = n // 2
                B0 = c0[0]
                B1 = c1[0]

                # solve Ax = B (no update clipping for 8x8, as before)
                ax += det_inv * (A11 * B0 - A01 * B1)
                ay += det_inv * (-A10 * B0 + A00 * B1)
            else:
                B0 = np.float32(0.0)
                B1 = np.float32(0.0)
                for ty in range(tile_size):
                    y = tile_y * tile_size + ty
                    for tx in range(tile_size):
                        x = tile_x * tile_size + tx
                        ## bilinear interpolation at (x + ax, y + ay), zero outside
                        floor_x = math.floor(x + ax)
                        floor_y = math.floor(y + ay)
                        frac_x = x + ax - floor_x
                        frac_y = y + ay - floor_y

                        if 0 <= floor_y < h and 0 <= floor_x < w:
                            m00 = moving[floor_y, floor_x]
                        else:
                            m00 = np.float32(0.0)
                        if 0 <= floor_y < h and 0 <= floor_x + 1 < w:
                            m01 = moving[floor_y, floor_x + 1]
                        else:
                            m01 = np.float32(0.0)
                        if 0 <= floor_y + 1 < h and 0 <= floor_x < w:
                            m10 = moving[floor_y + 1, floor_x]
                        else:
                            m10 = np.float32(0.0)
                        if 0 <= floor_y + 1 < h and 0 <= floor_x + 1 < w:
                            m11 = moving[floor_y + 1, floor_x + 1]
                        else:
                            m11 = np.float32(0.0)

                        lerpx_top = m00 + (m01 - m00) * frac_x
                        lerpx_bot = m10 + (m11 - m10) * frac_x
                        mov_interp = lerpx_top + (lerpx_bot - lerpx_top) * frac_y

                        gradt = mov_interp - ref_img[y, x]
                        B0 += -gradx[y, x] * gradt
                        B1 += -grady[y, x] * gradt

                # solve Ax = B, clipping the update step
                ax += min(max(det_inv * (A11 * B0 - A01 * B1), -clip_radius), clip_radius)
                ay += min(max(det_inv * (-A10 * B0 + A00 * B1), -clip_radius), clip_radius)

        alignment[tile_y, tile_x, 0] = ax
        alignment[tile_y, tile_x, 1] = ay
