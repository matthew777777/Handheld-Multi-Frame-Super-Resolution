# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:00:17 2022

This script contains :
    - The implementation of Algorithm 6: ComputeRobustness
    - The implementation of Algorithm 7: ComputeGuideImage
    - The implementation of Algorithm 8: ComputeLocalStatistics
    - The implementation of Algorithm 9: ComputeLocalMin


@author: jamyl
"""
import time

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from typing import Optional, Tuple

from .utils import getTime, DEFAULT_NUMPY_FLOAT_TYPE, clamp, round_half_away, timer
from .utils_image import dogson_biquadratic_kernel, dogson_quadratic_kernel
from .config import Config
from .debug_writer import DebugWriter


def init_robustness(ref_img: NDArray, config: Config):
    """
    Initialiazes the robustness etimation procedure by
    computing the local stats of the reference image

    Parameters
    ----------
    ref_img : Array[imshape_y, imshape_x]
        Raw reference image J_1
    config : Config
        parameters.

    Returns
    -------
    local_means : Array[imshape_y, imshape_x, channels]
        local means of the reference image.

    local_stds : Array[imshape_y, imshape_x, channels]
        local standard deviations of the reference image.

    """
    verbose_3 = config.verbose >= 3

    compute_guide_image_ = timer(compute_guide_image, verbose_3, " - Decimating images to RGB", ' - Image decimated')
    compute_local_stats_ = timer(compute_local_stats, verbose_3, end_s=' - Local stats estimated')

    imshape_y, imshape_x = ref_img.shape

    bayer_mode = (config.mode == 'bayer')

    # Computing guide image
    if bayer_mode:
        guide_ref_img = compute_guide_image_(ref_img)
    else:
        # Numba friendly code to add 1 channel
        guide_ref_img = ref_img.reshape((1, imshape_y, imshape_x))

    local_means, local_stds = compute_local_stats_(guide_ref_img)

    return local_means, local_stds


def compute_robustness(comp_img: NDArray, ref_local_means: NDArray, ref_local_var: NDArray,
                       flows: NDArray,
                       noise_model: Tuple[NDArray, NDArray], config: Config,
                       debug_writer: Optional[DebugWriter] = None) -> NDArray:
    """
    this is the implementation of Algorithm 6: ComputeRobustness
    Returns the robustnesses of the compared image J_n (n>1), based on the
    provided flow V_n(p) and the local statistics of the reference frame.

    Parameters
    ----------
    comp_img : Array[imsize_y, imsize_x]
        Compared raw image J_n (n>1).
    ref_local_means : Array[imsize_y, imsize_x, c]
        Local means of the reference image
    ref_local_stds : Array[imsize_y, imsize_x, c]
        Local standard deviations of the reference image
    flows : Array[n_patchs_y, n_patchs_y, 2]
        patch-wise optical flows of the compared image V_n(p)
    config : Config
        parameters.
    debug_writer : DebugWriter, optional
        Streams intermediate guide images to disk when provided.

    Returns
    -------
    r : Array[imsize_y, imsize_x]
        Locally minimized Robustness map, sampled at the center of
        every bayer quad
    """
    current_time, verbose_3 = time.perf_counter(), config.verbose >= 3

    compute_guide_image_ = timer(compute_guide_image, verbose_3, " - Decimating images to RGB", ' - Image decimated')
    compute_local_stats_ = timer(compute_local_stats, verbose_3, end_s=' - Local stats estimated')
    warp_stats_ = timer(warp_stats, verbose_3, end_s=' - Local stats warped and upscaled')
    compute_d_sigma_ = timer(compute_d_sigma, verbose_3, end_s=' - Estimated color distances')
    compute_s_ = timer(compute_s, verbose_3, end_s=' - Flow irregularities registered')
    robustness_threshold_ = timer(robustness_threshold, verbose_3, end_s=' - Robustness Estimated')
    local_min_ = timer(local_min, verbose_3, end_s=' - Robustness locally minimized')

    imshape_y, imshape_x = comp_img.shape

    bayer_mode = (config.mode == 'bayer')

    tile_size = config.alignment.tile_size
    assert isinstance(tile_size, int), f"Got invalide tile size {tile_size}"

    sigma_sq_curve, d_sq_curve = noise_model

    # Computing guide image
    if bayer_mode:
        guide_img = compute_guide_image_(comp_img)
    else:
        guide_img = comp_img.reshape((1, imshape_y, imshape_x)) # Adding 1 channel


    # Computing local stats (before applying optical flow)
    comp_local_means, _ = compute_local_stats_(guide_img)
    if debug_writer is not None:
        frame = np.moveaxis(comp_local_means, 0, -1)
        debug_writer.write_rgb("rgb_guides", frame)

    # Upscale and warp local means
    comp_local_means = warp_stats_(comp_local_means, tile_size, flows)
    if debug_writer is not None:
        frame = np.moveaxis(comp_local_means, 0, -1)
        debug_writer.write_rgb("rgb_guides_aligned", frame)

    # computing d_sq and sigma_sq (noise correction on the fly)
    d_sq, sigma_sq = compute_d_sigma_(ref_local_means, comp_local_means,
                                      ref_local_var, sigma_sq_curve, d_sq_curve,
                                      config.robustness.noise_correction)

    # applying flow discontinuity penalty
    S = compute_s_(flows, config.robustness.Mt, config.robustness.s1, config.robustness.s2)
    if debug_writer is not None:
        debug_writer.write_scalar("S", S, value_range=None)

    R = robustness_threshold_(d_sq, sigma_sq, S, config.robustness.t, tile_size, bayer_mode)
    if debug_writer is not None:
        debug_writer.write_scalar("R", R)

    r = local_min_(R)
    if debug_writer is not None:
        debug_writer.write_scalar("R_local_min", r)

    return r


def compute_guide_image(raw_img: NDArray):
    """
    This is the implementation of Algorithm 7: ComputeGuideImage
    Return the guide image G associated with the raw frame J

    Parameters
    ----------
    raw_img : Array[imshape_y, imshape_x]
        Raw frame J_n.

    Returns
    -------
    guide_img : Array[3, imshape_y//2, imshape_x//2]
        guide image.

    """
    imshape_y, imshape_x = raw_img.shape
    guide_imshape_y, guide_imshape_x = imshape_y//2, imshape_x//2
    guide_img = np.empty((3, guide_imshape_y, guide_imshape_x), DEFAULT_NUMPY_FLOAT_TYPE)

    cpu_compute_guide_image(raw_img, guide_img)

    return guide_img

@njit(parallel=True)
def cpu_compute_guide_image(raw_img, guide_img):
    _, h, w = guide_img.shape

    for ty in prange(h):
        for tx in range(w):
            guide_img[0, ty, tx] = np.sqrt(max(raw_img[2*ty, 2*tx], 0))
            guide_img[1, ty, tx] = np.sqrt(max(0.5*(raw_img[2*ty, 2*tx+1] + raw_img[2*ty+1, 2*tx]), 0))
            guide_img[2, ty, tx] = np.sqrt(max(raw_img[2*ty+1, 2*tx+1], 0))

def compute_local_stats(guide_img: NDArray):
    """
    Implementation of Algorithm 8: ComputeLocalStatistics
    Computes the mean color and variance associated for each 3 by 3 patches of
    the guide image G_n.

    Parameters
    ----------
    guide_img : Array[channels, guide_imshape_y, guide_imshape_x]
        Guide image G_n.

    Returns
    -------
    ref_local_means : Array[guide_imshape_y, guide_imshape_x, channels]
        Array that contains the local mean for every position of the guide image.
    ref_local_stds : Array[guide_imshape_y, guide_imshape_x, channels]
        Array that contains the local variance sigma² for every position of the guide image.


    """
    n_channels, *guide_imshape = guide_img.shape
    if n_channels == 1:
        mean = np.empty((1, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE)
        var = np.empty((1, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE)
    elif n_channels == 3:
        mean = np.empty((3, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE)
        var = np.empty((3, *guide_imshape), DEFAULT_NUMPY_FLOAT_TYPE)
    else:
        raise ValueError("Incoherent number of channel : {}".format(n_channels))

    cpu_compute_local_stats(guide_img, mean, var)

    return mean, var


@njit(parallel=True)
def cpu_compute_local_stats(guide_img, mean, var):
    n_channels, guide_imshape_y, guide_imshape_x = guide_img.shape

    for idx in prange(n_channels * guide_imshape_y * guide_imshape_x):
        channel = idx // (guide_imshape_y * guide_imshape_x)
        rem = idx % (guide_imshape_y * guide_imshape_x)
        idy = rem // guide_imshape_x
        x = rem % guide_imshape_x

        mean_ = np.float32(0)
        var_ = np.float32(0)
        for i in range(-1, 2):
            for j in range(-1, 2):
                y = clamp(idy + i, 0, guide_imshape_y-1)
                xx = clamp(x + j, 0, guide_imshape_x-1)

                color = guide_img[channel, y, xx]
                mean_ += color
                var_ += color * color

        # normalizing
        mean_ /= 9
        mean[channel, idy, x] = mean_
        var[channel, idy, x] = var_ / 9 - mean_ * mean_

def warp_stats(local_stats: NDArray, tile_size: int, flow: NDArray):
    """
    Upscales and warps a map of local statistics using Dogson's biquadratic approximation

    Parameters
    ----------
    local_stats : array [guide_imshape_y, guide_imshape_x, n_c]
        A map of ONE local stat (can have 1 or 3 channels)
    tile_size : Integer
        If required, flow tile size.
    flow : Array [ty, tx, 2], optional
        If required, the optical flow. The default is None.

    Returns
    -------
    upscaled_stats : Array[raw_imshape_y, raw_imshape_y, c]
        Upscaled and warped local stats

    """
    n_channels, *guide_imshape = local_stats.shape
    bayer_mode = (n_channels == 3)

    warped_stats = np.empty((n_channels,
                             guide_imshape[0],
                             guide_imshape[1]),
                            DEFAULT_NUMPY_FLOAT_TYPE)

    cpu_warp_dogson(local_stats, flow, tile_size, warped_stats)
    return warped_stats


@njit(parallel=True)
def cpu_warp_dogson(source, flow, tile_size, warped):
    tile_size = tile_size // 2
    n_channels, ny, nx = source.shape

    for pix_id in prange(ny * nx):
        y = pix_id // nx
        x = pix_id % nx

        # Flow is defined on the raw image basis
        patch_idy = int(y//tile_size)
        patch_idx = int(x//tile_size)

        flow_x = flow[patch_idy, patch_idx, 0]
        flow_y = flow[patch_idy, patch_idx, 1]


        # Jumping from ref guide to mov guide
        y_mov = y + flow_y * 0.5
        x_mov = x + flow_x * 0.5

        # Out of bounds
        if not (0 <= y_mov < ny and
                0 <= x_mov < nx):
            for c in range(n_channels):
                warped[c, y, x] = np.inf # infinity will imply R = 0
            continue

        center_y = round_half_away(y_mov)
        center_x = round_half_away(x_mov)

        # init buffer
        w_acc = np.float32(0)
        buffer = np.empty(3, np.float32)
        for c in range(n_channels):
            buffer[c] = 0

        for i in range(-1, 2):
            y_ = int(clamp(center_y + i, 0, ny-1))
            dy = y_ - y_mov
            wy = dogson_quadratic_kernel(dy)
            for j in range(-1, 2):
                x_ = int(clamp(center_x + j, 0, nx-1))
                dx = x_ - x_mov

                w = wy * dogson_quadratic_kernel(dx)

                for c in range(n_channels):
                    buffer[c] += source[c, y_, x_] * w
                w_acc += w

        # Normalise and write output
        for c in range(n_channels):
            warped[c, y, x] = buffer[c] / w_acc


def compute_d_sigma(means_r: NDArray, means_m: NDArray, var_r: NDArray, sigma_sq_curve: NDArray, d_sq_curve: NDArray, do_noise_correction: bool):
    """
    Computes the color distance between the two frames. They must be warped.

    Parameters
    ----------
    means_1 : array [ny, nx, c]
        local mean of frame 1.
    means_2 : array [ny, nx, c]
        local mean of frame 1.

    Returns
    -------
    diff : array [ny, nx, c]
        channel wise absolute difference

    """
    assert means_r.shape == means_m.shape
    nc, ny, nx = shape = means_r.shape
    d_sq = np.empty((ny, nx), DEFAULT_NUMPY_FLOAT_TYPE)
    sigma_sq = np.empty((ny, nx), DEFAULT_NUMPY_FLOAT_TYPE)

    cpu_compute_d_sigma(means_r, means_m, var_r, sigma_sq_curve, d_sq_curve, d_sq, sigma_sq, do_noise_correction)

    return d_sq, sigma_sq

@njit(parallel=True)
def cpu_compute_d_sigma(means_r, means_m, var_r, sigma_sq_curve, d_sq_curve, d_sq, sigma_sq, do_noise_correction):
    nc, ny, nx = means_r.shape

    for pix_id in prange(ny * nx):
        y = pix_id // nx
        x = pix_id % nx

        d_sq_ = np.float32(0)
        sigma_sq_ = np.float32(0)
        for c in range(nc):
            error = means_r[c, y, x] - means_m[c, y, x]
            d_sq_ += error * error
            sigma_sq_ += var_r[c, y, x]


        if do_noise_correction:
            brightness = np.float32(0)
            for c in range(nc):
                brightness += means_r[c, y, x]
            brightness /= nc
            brightness = clamp(brightness, 0, 1)
            id_noise = round_half_away((sigma_sq_curve.shape[0] - 1) * brightness)

            d_noise_sq = d_sq_curve[id_noise]
            sigma_noise_sq = sigma_sq_curve[id_noise]
            sigma_sq_ = max(sigma_sq_, sigma_noise_sq)

            if d_sq_ > 0:
                shrink = d_sq_ / (d_sq_ + d_noise_sq)
                d_sq_ *= shrink * shrink

        d_sq[y, x] = d_sq_
        sigma_sq[y, x] = sigma_sq_


def compute_s(flows: NDArray, M_th: float, s1: float, s2: float):
    """ Computes s at every position based on flow irregularities


    Parameters
    ----------
    flows : Array[n_tiles_y, n_tiles_x, 2]
        Patch wise optical flow
    M_th : float
        Threshold for M.
    s1 : float
        DESCRIPTION.
    s2 : float
        DESCRIPTION.

    Returns
    -------
    S : Array[n_patchs_y, n_patchs_x]
        Map where s1 or s2 will be written at each position.

    """
    n_patch_y, n_patch_x, _ = flows.shape
    S = np.empty((n_patch_y, n_patch_x), DEFAULT_NUMPY_FLOAT_TYPE)

    cpu_compute_s(flows, M_th, s1, s2, S)

    return S

@njit(parallel=True)
def cpu_compute_s(flows, M_th, s1, s2, S):
    n_patch_y, n_patch_x, _ = flows.shape

    for patch_id in prange(n_patch_y * n_patch_x):
        patch_idy = patch_id // n_patch_x
        patch_idx = patch_id % n_patch_x

        mini0 = np.float32(np.inf)
        mini1 = np.float32(np.inf)
        maxi0 = np.float32(-np.inf)
        maxi1 = np.float32(-np.inf)

        for i in range(-1, 2):
            for j in range(-1, 2):
                y = patch_idy + i
                x = patch_idx + j

                inbound = (0 <= x < n_patch_x and
                           0 <= y < n_patch_y)

                if inbound:
                    flow0 = flows[y, x, 0]
                    flow1 = flows[y, x, 1]

                    #local max search
                    maxi0 = max(maxi0, flow0)
                    maxi1 = max(maxi1, flow1)
                    #local min search
                    mini0 = min(mini0, flow0)
                    mini1 = min(mini1, flow1)

        diff_0 = maxi0 - mini0
        diff_1 = maxi1 - mini1
        if diff_0*diff_0 + diff_1*diff_1 > M_th*M_th:
            S[patch_idy, patch_idx] = s1
        else:
            S[patch_idy, patch_idx] = s2

def robustness_threshold(d_sq: NDArray, sigma_sq: NDArray, S: NDArray, t: float, tile_size: int, bayer_mode: bool):
    imshape = d_sq.shape
    R = np.empty(imshape, DEFAULT_NUMPY_FLOAT_TYPE)

    cpu_robustness_threshold(d_sq, sigma_sq, S, t, tile_size, bayer_mode, R)

    return R

@njit(parallel=True)
def cpu_robustness_threshold(d_sq, sigma_sq, S, t, tile_size, bayer_mode, R):
    tile_size = tile_size//2

    ny, nx = R.shape
    for pix_id in prange(ny * nx):
        idy = pix_id // nx
        idx = pix_id % nx

        patch_idy = int(idy//tile_size)
        patch_idx = int(idx//tile_size)

        R[idy, idx] = clamp(S[patch_idy, patch_idx] * np.exp(-d_sq[idy, idx]/sigma_sq[idy, idx]) - t,
                            0, 1)

def local_min(R: NDArray):
    """
    Implementation of Algorithm 9: ComputeLocalMin
    For each pixel of R, the minimum in a 5 by 5 window is estimated
    and stored in r.

    Parameters
    ----------
    R : Array[guide_imshape_y, guide_imshape_x]
        Robustness map for every image

    Returns
    -------
    r : Array[guide_imshape_y, guide_imshape_x]
        locally minimised version of R

    """
    r = np.empty(R.shape, DEFAULT_NUMPY_FLOAT_TYPE)

    cpu_compute_local_min(R, r)

    return r

@njit(parallel=True)
def cpu_compute_local_min(R, r):
    guide_imshape_y, guide_imshape_x = R.shape

    for pix_id in prange(guide_imshape_y * guide_imshape_x):
        idy = pix_id // guide_imshape_x
        idx = pix_id % guide_imshape_x

        mini = np.float32(np.inf)

        #local min search
        for i in range(-2, 3):
            y = clamp(idy + i, 0, guide_imshape_y-1)
            for j in range(-2, 3):
                x = clamp(idx + j, 0, guide_imshape_x-1)
                mini = min(mini, R[y, x])

        r[idy, idx] = mini
