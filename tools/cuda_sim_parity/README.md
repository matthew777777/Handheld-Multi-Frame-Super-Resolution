# CUDA parity of the CPU port, without an NVIDIA GPU

Runs the original Numba-CUDA kernels (Jamy-L upstream 27c4655, the parent of
the CPU port) unmodified under the Numba CUDA simulator and compares them
kernel by kernel with `handheld_super_resolution` (this CPU port).

    tools/cuda_sim_parity/setup.sh
    NUMBA_ENABLE_CUDASIM=1 python tools/cuda_sim_parity/kernel_parity.py       # ~15 min on an M1
    NUMBA_ENABLE_CUDASIM=1 python tools/cuda_sim_parity/tile64_known_shift.py

`simshim.py` maps torch tensors to simulator device arrays (zero copy, as
`__cuda_array_interface__` does on a GPU) and emulates `cuda.shfl_down_sync`,
which the simulator lacks. The simulator evaluates with Python/NumPy-2
promotion rules, not compiled Numba-CUDA typing, so differences around 1e-5
are typing noise; it is a logic oracle.

## Results (2026-11-14)

All kernels agree to <= 4e-5 (ICA 8/16/32 and L1 16/32 within 1e-6 or exact)
after the fixes in this branch:

- `round()` in the CUDA kernels is round-half-to-even (numba.cuda lowers it
  to `__nv_llrint`), not half-away. Matters for the Dogson warp at the image
  border with exact .5 positions (was 0.016 in the warped means, 0.31 in R),
  and at exact ties for the L1 start flow and noise-LUT index.
- The CUDA kernels' int-initialised accumulators (local stats, warp weight
  sum, d/sigma) are float64 after Numba type unification; the port now
  matches.

Two intentional deviations remain, both in the 64 px tile path used when
SNR <= 14 dB, where the original CUDA kernels are wrong:

- `cuda_L1_local_search64` never applies the candidate shift when reading
  the moving patch, so it always returns (-R, -R)
  (`tile64_known_shift.py`: true (2,-1), (-3,2), (0,0) -> CUDA (-4,-4) each).
- `ica_kernel_64` interpolates rows y0 and y0+2 instead of y0 and y0+1
  (known 0.4 px shift: CUDA y -1.063, port -0.147; x identical).
