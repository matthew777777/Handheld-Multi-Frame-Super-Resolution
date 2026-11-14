"""Known-shift check of the 64 px tile kernels: original CUDA (simulator) vs CPU port.
Run: tools/cuda_sim_parity/setup.sh, then NUMBA_ENABLE_CUDASIM=1 python tools/cuda_sim_parity/tile64_known_shift.py"""
import os, sys
H = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(H, "work")
sys.path[:0] = [H, os.path.join(W, "kp"), os.path.join(W, "upsim")]
import simshim, numpy as np, torch  # noqa
from hsr_up import ICA as Ui, block_matching as Ub
from hsr_fork import ICA as Fi, block_matching as Fb
from hsr_fork.config import Config
rng = np.random.default_rng(3); S = 128
yy, xx = np.mgrid[0:S, 0:S]
ref = (0.5 + 0.3*np.sin(xx/5.0)*np.cos(yy/7.0)).astype(np.float32)
for true in [(2, -1), (-3, 2), (0, 0)]:
    mov = np.roll(ref, (true[1], true[0]), (0, 1)).astype(np.float32)   # content moves by +true -> flow = +true
    c = Config(); c.alignment.tile_sizes = [64]; c.alignment.search_radii = [4]
    au, af = torch.zeros((2, 2, 2)), torch.zeros((2, 2, 2))
    Ub.align_lvl_block_matching_L1(torch.as_tensor(ref), torch.as_tensor(mov), au, 0, c)
    Fb.align_lvl_block_matching_L1(torch.as_tensor(ref), torch.as_tensor(mov), af, 0, c)
    print(f"L1 ts=64 true flow {true}: CUDA -> {au[0,0].tolist()}  fork -> {af[0,0].tolist()}")
sub = 0.4
mov = np.ascontiguousarray((0.5 + 0.3*np.sin((xx+sub)/5.0)*np.cos((yy+sub)/7.0)).astype(np.float32))
for ts in (32, 64):
    c = Config(); c.alignment.tile_sizes = [ts]; c.alignment.search_radii = [4]; c.alignment.ica.n_iter = 3
    g, gm = torch.as_tensor(ref), torch.as_tensor(mov)
    gxu, gyu, hu = Ui.init_ica(g, ts, c); gxf, gyf, hf = Fi.init_ica(g, ts, c)
    au, af = torch.zeros((S//ts, S//ts, 2)), torch.zeros((S//ts, S//ts, 2))
    Ui.align_lvl_ica(g, gxu, gyu, hu, gm, au, 0, c); Fi.align_lvl_ica(g, gxf, gyf, hf, gm, af, 0, c)
    print(f"ICA ts={ts} true flow ({sub},{sub}): CUDA -> {np.round(au[1,1].numpy(),3).tolist()}  fork -> {np.round(af[1,1].numpy(),3).tolist()}")
