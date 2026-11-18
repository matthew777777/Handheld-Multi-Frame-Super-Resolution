#!/bin/sh
# Prepares the original Numba-CUDA code (Jamy-L upstream, commit b71c534, the
# upstream state the CPU port is merged with; alignment is native Halide in both) so it runs under the Numba CUDA simulator on a
# machine without an NVIDIA GPU, next to this CPU port:
#   work/upsim/handheld_super_resolution   upstream, torch device "cuda" -> "cpu"
#   work/kp/hsr_up, work/kp/hsr_fork       both packages under distinct names
# Then:  NUMBA_ENABLE_CUDASIM=1 python tools/cuda_sim_parity/kernel_parity.py
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); W="$HERE/work"
mkdir -p "$W/kp"
[ -d "$W/upsim" ] || git clone -q https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution "$W/upsim"
git -C "$W/upsim" checkout -q b71c5344e40a4a738ac6494f3b6ec5897922e29b
sed -i.bak 's/device="cuda"/device="cpu"/g' "$W"/upsim/handheld_super_resolution/*.py && rm -f "$W"/upsim/handheld_super_resolution/*.bak
ln -sfn "$W/upsim/handheld_super_resolution" "$W/kp/hsr_up"
ln -sfn "$ROOT/handheld_super_resolution" "$W/kp/hsr_fork"
echo "ready: NUMBA_ENABLE_CUDASIM=1 python $HERE/kernel_parity.py"
