"""Kernel-by-kernel parity: original Numba-CUDA kernels (CUDA simulator) vs the fork's CPU ports.
Run: tools/cuda_sim_parity/setup.sh, then NUMBA_ENABLE_CUDASIM=1 python tools/cuda_sim_parity/kernel_parity.py"""
import os, sys
assert os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
H = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(H, "work")
sys.path[:0] = [H, os.path.join(W, "kp"), os.path.join(W, "upsim")]
import simshim  # noqa
import numpy as np, torch
from numba import cuda
import hsr_up, hsr_fork  # noqa
from hsr_up import utils_image as Uu, robustness as Ur, kernels as Uk, merge as Um
from hsr_fork import utils_image as Fu, robustness as Fr, kernels as Fk, merge as Fm
from hsr_fork.config import Config
from hsr_fork.params import update_snr_config

def host(x):
    if hasattr(x, "copy_to_host"): return x.copy_to_host()
    if isinstance(x, torch.Tensor): return x.detach().numpy()
    return np.asarray(x)
ROWS = []
def check(name, a, b, tol=0.0):
    a, b = host(a).astype(np.float64), host(b).astype(np.float64)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    nm = (np.isnan(a) != np.isnan(b)).sum(); m = ~(np.isnan(a) | np.isnan(b))
    fin = m & np.isfinite(a) & np.isfinite(b); im = (np.isinf(a[m]) != np.isinf(b[m])).sum()
    d = np.abs(a - b)[fin]; mx = d.max() if d.size else 0.0
    rel = (d / np.maximum(np.abs(a[fin]), 1e-12)).max() if d.size else 0.0
    exact = int((d == 0).mean() * 1000) / 10 if d.size else 100.0
    ok = nm == 0 and im == 0 and mx <= tol
    ROWS.append((name, a.shape, mx, rel, exact, nm, im, ok))
    print(f"{'OK ' if ok else 'DIFF'} {name:34s} {str(a.shape):18s} max|d|={mx:.3g} maxrel={rel:.3g} exact={exact}% nan-mis={nm} inf-mis={im}", flush=True)

rng = np.random.default_rng(0)
S = 64
raw = np.clip(0.3 + 0.2*np.sin(np.arange(S)[None]/3.0)*np.cos(np.arange(S)[:, None]/4.0) + 0.02*rng.standard_normal((S, S)), 0, 1).astype(np.float32)
raw[10:16, 20:30] = 1.0  # clipped highlight: flat -> zero gradients -> NaN path in compute_k
mov = np.roll(raw, (1, -2), (0, 1)).astype(np.float32) + 0.01*rng.standard_normal((S, S)).astype(np.float32)
alpha, beta = (0.0012, 0.00126, 0.0012, 0.00126), (4e-7, 4e-7, 3.8e-7, 4e-7)

check("gat", Uu.gat(cuda.to_device(raw), alpha, beta), Fu.gat(raw, alpha, beta), 1e-5)
check("bayer quad average", Uu.average_bayer_quads(cuda.to_device(raw)), Fu.average_bayer_quads(raw), 1e-7)
from hsr_fork.config import GrayscaleMethod
from hsr_up.config import GrayscaleMethod as UGM
check("alignment grey (decimating)", Uu.compute_grey_images(raw, UGM.DECIMATING), Fu.compute_grey_images(raw, GrayscaleMethod.DECIMATING))
gu, gf = Ur.compute_guide_image(cuda.to_device(raw)), Fr.compute_guide_image(raw); check("guide image", gu, gf)
mu, vu = Ur.compute_local_stats(gu); mf, vf = Fr.compute_local_stats(gf)
check("local means", mu, mf, 1e-7); check("local vars", vu, vf, 1e-7)
flow = (rng.standard_normal((3, 3, 2)) * 1.5)  # 16px/stride-8 grid on the 32x32 grey.astype(np.float32); flow[0, 0] = (1.0, -1.0)  # odd ints -> .5 positions
gm_u, gm_f = Ur.compute_guide_image(cuda.to_device(mov)), Fr.compute_guide_image(mov)
mmu, _ = Ur.compute_local_stats(gm_u); mmf, _ = Fr.compute_local_stats(gm_f)
wu, wf = Ur.warp_stats(mmu, 16, cuda.to_device(flow)), Fr.warp_stats(mmf, 16, flow); check("warp dogson", wu, wf, 1e-6)
curve_s = np.linspace(1e-4, 4e-3, 65).astype(np.float32); curve_d = np.linspace(2e-4, 8e-3, 65).astype(np.float32)
for nc in (False, True):
    du, su = Ur.compute_d_sigma(mu, wu, vu, cuda.to_device(curve_s), cuda.to_device(curve_d), nc)
    df, sf = Fr.compute_d_sigma(mf, wf, vf, curve_s, curve_d, nc)
    check(f"d_sq noise_corr={nc}", du, df, 1e-6); check(f"sigma_sq noise_corr={nc}", su, sf, 1e-6)
Su, Sf = Ur.compute_s(cuda.to_device(flow), 0.8, 2.0, 12.0), Fr.compute_s(flow, 0.8, 2.0, 12.0); check("S (motion prior)", Su, Sf)
Ru, Rf = Ur.robustness_threshold(du, su, Su, 0.12, 16, True), Fr.robustness_threshold(df, sf, Sf, 0.12, 16, True); check("R threshold", Ru, Rf, 1e-6)
check("local min 5x5", Ur.local_min(Ru), Fr.local_min(Rf), 1e-6)

c = Config(); c.verbose = 0; c.noise_model.alpha, c.noise_model.beta = alpha, beta
update_snr_config(c, 18.0)
for law in ("linear", "hard_threshold"):
    c.merging.selection_law = law
    check(f"kernels covs ({law})", Uk.estimate_kernels(cuda.to_device(raw), c), Fk.estimate_kernels(raw, c), 1e-5)
c.merging.selection_law = "linear"
covs_u, covs_f = Uk.estimate_kernels(cuda.to_device(raw), c), Fk.estimate_kernels(raw, c)
covs_f = host(covs_u).copy()  # feed identical covariances to both merges
r = np.clip(rng.random((S//2, S//2)), 0, 1).astype(np.float32)
flow_m = (rng.standard_normal((4, 4, 2)) * 1.5).astype(np.float32)
for scale in (1, 2, 1.5):
    c.scale = scale; c.alignment.tile_size = 16
    oh, ow = round(scale*S), round(scale*S)
    nu, du_ = cuda.to_device(np.zeros((oh, ow, 3), np.float32)), cuda.to_device(np.zeros((oh, ow, 3), np.float32))
    nf, df_ = np.zeros((oh, ow, 3), np.float32), np.zeros((oh, ow, 3), np.float32)
    # 4x4 grid: upstream merge does not clamp the tile index (reads one tile
    # past a 3x3 grid at the border); the port clamps, so compare in-range.
    Um.merge(cuda.to_device(mov), cuda.to_device(flow_m), covs_u, cuda.to_device(r), nu, du_, c)
    Fm.merge(mov, flow_m, covs_f, r, nf, df_, c)
    check(f"merge num scale={scale}", nu, nf, 1e-6); check(f"merge den scale={scale}", du_, df_, 1e-6)

print(f"\n{sum(r[-1] for r in ROWS)}/{len(ROWS)} within tolerance")
