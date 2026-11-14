"""Numba CUDA simulator shims so the original kernels run unmodified on CPU-only machines."""
"""Run upstream Numba-CUDA code on CPU via NUMBA_ENABLE_CUDASIM=1.
torch tensors <-> simulator device arrays share memory (no copies)."""
import os
assert os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
import numpy as np, torch
from numba import cuda
from numba.cuda.simulator.cudadrv.devicearray import FakeCUDAArray
def _as_cuda_array(t, sync=True):
    if isinstance(t, FakeCUDAArray): return t
    return FakeCUDAArray(t.detach().numpy())
cuda.as_cuda_array = _as_cuda_array
_orig = torch.as_tensor
def _as_tensor(x, *a, **k):
    if isinstance(x, FakeCUDAArray): x = x._ary
    return _orig(x, *a, **k)
torch.as_tensor = _as_tensor
import numba.cuda.simulator.cudadrv.devicearray as _sda
_sda.DeviceNDArray = FakeCUDAArray
import sys as _s, types as _t
_s.modules.setdefault("numba.cuda.cudadrv.devicearray", _sda)
# Real numba-cuda accepts torch CUDA tensors via __cuda_array_interface__ (zero copy).
# Mirror that: torch CPU tensors become simulator device arrays sharing memory.
from numba.cuda.simulator import kernel as _k
_orig_call = _k.FakeCUDAKernel.__call__
def _call(self, *args):
    args = tuple(FakeCUDAArray(a.detach().numpy()) if isinstance(a, torch.Tensor) else a for a in args)
    return _orig_call(self, *args)
_k.FakeCUDAKernel.__call__ = _call
# Warp shuffle for the simulator (absent there). Semantics of __shfl_down_sync with a
# full mask: lane l reads lane l+delta of its warp; lanes past the warp end keep their
# own value. Implemented with two block barriers (all threads of these kernels call it).
import threading as _th
from numba.cuda.simulator import kernelapi as _ka
def _shfl_down_sync(self, mask, value, delta):
    t = _th.current_thread(); m = t._manager
    if not hasattr(m, "_shfl"): m._shfl = {}
    m._shfl[t.thread_id] = value
    t.syncthreads()
    lane = t.thread_id % 32
    out = m._shfl[t.thread_id + delta] if lane + delta < 32 and (t.thread_id + delta) in m._shfl else value
    t.syncthreads()
    return type(value)(out) if isinstance(value, np.floating) else out
_ka.FakeCUDAModule.shfl_down_sync = _shfl_down_sync
