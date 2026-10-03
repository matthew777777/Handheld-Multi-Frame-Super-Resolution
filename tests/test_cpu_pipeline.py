"""CPU regression tests for the handheld super-resolution pipeline.

Runs on stock NumPy/torch/Numba CPU builds (no GPU required):

    python -m unittest discover tests -v
"""
import importlib.util
import os
import tempfile
import unittest

import numpy as np

from handheld_super_resolution.block_matching import align_lvl_block_matching_L1
from handheld_super_resolution.config import Config
from handheld_super_resolution.noise_lut import NoiseLut, save_noise_lut
from handheld_super_resolution.params import sanitize_config, update_snr_config
from handheld_super_resolution.super_resolution import main, process
from handheld_super_resolution.utils_image import gat

HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None

SIZE = 512
ALPHA = (1e-4,) * 4
BETA = (1e-6,) * 4


def synthetic_burst(size=SIZE, n_frames=3, seed=0):
    """Smooth Bayer-like burst with small integer shifts between frames."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = (np.sin(xx / 12) * np.cos(yy / 15) * 0.5 + 0.5).astype(np.float32)
    base[0::2, 0::2] *= 0.8
    base[1::2, 1::2] *= 0.6
    frames = []
    for i in range(n_frames):
        shifted = np.roll(base, shift=(i, -i), axis=(0, 1))
        noisy = np.clip(shifted + rng.normal(0, 0.01, size=shifted.shape), 0, 1)
        frames.append(noisy.astype(np.float32))
    return np.stack(frames)


def test_config(**overrides):
    config = Config()
    config.verbose = 0
    config.scale = 2
    config.force_snr = 24.0
    config.robustness.noise_correction = False
    config.noise_model.alpha = ALPHA
    config.noise_model.beta = BETA
    config.noise_model.sigma_sq_curve = [0.0, 0.0]
    config.noise_model.d_sq_curve = [0.0, 0.0]
    for key, value in overrides.items():
        setattr(config, key, value)
    update_snr_config(config, config.force_snr)
    sanitize_config(config, (SIZE, SIZE))
    return config


def write_synthetic_dng(path, data, black=64, white=4095):
    """Write a minimal uncompressed RGGB DNG readable by rawpy/exifread."""
    import tifffile

    sr_one = (10000, 10000)
    sr_zero = (0, 10000)
    color_matrix = [sr_one if i % 4 == 0 else sr_zero for i in range(9)]
    extratags = [
        (271, "s", 0, "TestCam", True),                       # Make
        (272, "s", 0, "TestCam 1", True),                     # Model
        (274, "H", 1, 1, True),                               # Orientation
        (34855, "H", 1, 100, True),                           # ISOSpeedRatings
        (50706, "B", 4, bytes((1, 4, 0, 0)), True),           # DNGVersion
        (50707, "B", 4, bytes((1, 0, 0, 0)), True),           # DNGBackwardVersion
        (33421, "H", 2, (2, 2), True),                        # CFARepeatPatternDim
        (33422, "B", 4, bytes((0, 1, 1, 2)), True),           # CFAPattern RGGB
        (50711, "H", 1, 1, True),                             # CFALayout
        (50714, "H", 1, black, True),                         # BlackLevel
        (50717, "H", 1, white, True),                         # WhiteLevel
        (50721, "2i", 9, color_matrix, True),                 # ColorMatrix1
        (50778, "H", 1, 21, True),                            # CalibrationIlluminant1 = D65
        (50728, "2I", 3, [(1000, 1000)] * 3, True),           # AsShotNeutral
        (51041, "d", 6, [ALPHA[0], BETA[0]] * 3, True),       # NoiseProfile
    ]
    tifffile.imwrite(
        path, data, extratags=extratags,
        compression=1, planarconfig=1, predictor=False,
        photometric=32803,  # Color Filter Array
    )


class TestCpuPipeline(unittest.TestCase):
    def test_main_bayer_end_to_end(self):
        burst = synthetic_burst()
        output, debug = main(burst[0], burst[1:], test_config())
        self.assertEqual(output.shape, (SIZE * 2, SIZE * 2, 3))
        self.assertEqual(output.dtype, np.float32)
        # Border pixels may be NaN where no frame contributes (as on CUDA),
        # but the interior must be fully reconstructed.
        self.assertTrue(np.isfinite(output[8:-8, 8:-8]).all())
        self.assertIn("accumulated robustness", debug)
        self.assertEqual(
            debug["accumulated robustness"].shape, (SIZE // 2, SIZE // 2)
        )

    def test_main_grey_channel_zero(self):
        burst = synthetic_burst()
        output, _ = main(burst[0], burst[1:], test_config(mode="grey"))
        self.assertEqual(output.shape, (SIZE * 2, SIZE * 2, 3))
        # Grey mode only accumulates channel 0 (as on CUDA).
        self.assertTrue(np.isfinite(output[8:-8, 8:-8, 0]).all())

    def test_l1_block_matching_recovers_shift(self):
        import torch

        rng = np.random.default_rng(1)
        ref = rng.random((64, 64), dtype=np.float32)
        moving = np.roll(ref, shift=(3, -2), axis=(0, 1))
        alignments = torch.zeros((4, 4, 2), dtype=torch.float32)
        config = Config()
        config.alignment.tile_sizes = [16]
        config.alignment.search_radii = [4]
        align_lvl_block_matching_L1(
            torch.from_numpy(ref), torch.from_numpy(moving),
            alignments, 0, config,
        )
        # np.roll shift (dy, dx) corresponds to flow (dx, dy).
        np.testing.assert_array_equal(
            alignments.numpy()[1:3, 1:3, 0], np.full((2, 2), -2.0)
        )
        np.testing.assert_array_equal(
            alignments.numpy()[1:3, 1:3, 1], np.full((2, 2), 3.0)
        )

    def test_gat_matches_numpy(self):
        image = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        alpha = (0.1, 0.2, 0.3, 0.4)
        beta = (0.01, 0.02, 0.03, 0.04)
        got = gat(image, alpha, beta)
        # EXIF plane order is R, G1, B, G2; mosaic layout is R G1 / G2 B.
        planes = np.empty(image.shape, dtype=np.intp)
        planes[0::2, 0::2] = 0
        planes[0::2, 1::2] = 1
        planes[1::2, 1::2] = 2
        planes[1::2, 0::2] = 3
        alpha_arr = np.vectorize(alpha.__getitem__)(planes)
        beta_arr = np.vectorize(beta.__getitem__)(planes)
        vst = np.maximum(alpha_arr * image + 3 / 8 * alpha_arr**2 + beta_arr, 0)
        expected = (2 / alpha_arr * np.sqrt(vst)).astype(np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    @unittest.skipUnless(HAS_TIFFFILE, "tifffile required to write test DNGs")
    def test_process_dng_burst_end_to_end(self):
        black, white = 64, 4095
        base = synthetic_burst(n_frames=1, seed=2)[0]
        raw = np.clip(base * (white - black - 500) + black + 250, 0, white)
        raw = raw.astype(np.uint16)
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (dy, dx) in enumerate([(0, 0), (0, 2), (2, 0)]):
                write_synthetic_dng(
                    os.path.join(tmpdir, f"frame_{i:02d}.dng"),
                    np.roll(raw, shift=(dy, dx), axis=(0, 1)),
                )
            # Crafted LUT covering the noise-correction branch.
            bins = 8
            brightness = np.linspace(0.0, 1.0, bins, dtype=np.float32)
            lut = NoiseLut(
                brightness=brightness,
                sigma_noise_sq=np.full(bins, 1e-4, dtype=np.float32),
                d_noise_sq=np.full(bins, 1e-4, dtype=np.float32),
                bin_counts=np.full(bins, 10, dtype=np.int64),
                sigma_noise_sq_sem=np.full(bins, 1e-5, dtype=np.float32),
                d_noise_sq_sem=np.full(bins, 1e-5, dtype=np.float32),
                alpha=np.asarray(ALPHA, dtype=np.float64),
                beta=np.asarray(BETA, dtype=np.float64),
            )
            lut_path = os.path.join(tmpdir, "test_noise.npz")
            save_noise_lut(lut_path, lut, trials=bins * 10, seed=0)

            config = Config()
            config.verbose = 0
            config.scale = 2
            config.force_snr = 24.0
            config.noise_model.lut_path = lut_path
            output, _ = process(tmpdir, config)

        self.assertEqual(output.shape, (SIZE * 2, SIZE * 2, 3))
        self.assertTrue(np.isfinite(output).all())
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    @unittest.skipUnless(HAS_TIFFFILE, "tifffile required to write test DNGs")
    def test_process_debug_mode(self):
        black, white = 64, 4095
        base = synthetic_burst(n_frames=1, seed=3)[0]
        raw = np.clip(base * (white - black - 500) + black + 250, 0, white)
        raw = raw.astype(np.uint16)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            burst_dir = os.path.join(tmpdir, "burst")
            os.makedirs(burst_dir)
            for i in range(2):
                write_synthetic_dng(
                    os.path.join(burst_dir, f"frame_{i:02d}.dng"), raw
                )
            os.chdir(tmpdir)  # debug frames land in a temp working copy
            try:
                config = Config()
                config.verbose = 0
                config.scale = 1
                config.debug = True
                config.force_snr = 24.0
                config.robustness.noise_correction = False
                output, _ = process(burst_dir, config)
            finally:
                os.chdir(cwd)
            self.assertEqual(output.shape, (SIZE, SIZE, 3))
            self.assertTrue(np.isfinite(output).all())
            debug_root = os.path.join(tmpdir, "debug")
            self.assertTrue(os.path.isdir(debug_root))
            run_dirs = os.listdir(debug_root)
            self.assertEqual(len(run_dirs), 1)
            categories = os.listdir(os.path.join(debug_root, run_dirs[0]))
            self.assertIn("optical_flow", categories)
            self.assertIn("R", categories)


if __name__ == "__main__":
    unittest.main()
