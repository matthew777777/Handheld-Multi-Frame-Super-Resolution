#!/bin/sh
# Prepares the original Numba-CUDA code (Jamy-L upstream, commit 27c4655, the
# parent of the CPU port) so it runs under the Numba CUDA simulator on a
# machine without an NVIDIA GPU, next to this CPU port:
#   work/upsim/handheld_super_resolution   upstream, torch device "cuda" -> "cpu"
#   work/kp/hsr_up, work/kp/hsr_fork       both packages under distinct names
# Then:  NUMBA_ENABLE_CUDASIM=1 python tools/cuda_sim_parity/kernel_parity.py
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd); W="$HERE/work"
mkdir -p "$W/kp"
[ -d "$W/upsim" ] || git clone -q https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution "$W/upsim"
git -C "$W/upsim" checkout -q 27c4655ae4752898e0ae6e9737697c4d89c95226
sed -i.bak 's/device="cuda"/device="cpu"/g' "$W"/upsim/handheld_super_resolution/*.py && rm -f "$W"/upsim/handheld_super_resolution/*.bak
ln -sfn "$W/upsim/handheld_super_resolution" "$W/kp/hsr_up"
ln -sfn "$ROOT/handheld_super_resolution" "$W/kp/hsr_fork"
echo "ready: NUMBA_ENABLE_CUDASIM=1 python $HERE/kernel_parity.py"
