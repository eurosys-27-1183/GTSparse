#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
backend="${1:?usage: ./run_e2e_v2.sh <backend> <dtype> [frames] [warmup_batches]}"
dtype="${2:?usage: ./run_e2e_v2.sh <backend> <dtype> [frames] [warmup_batches]}"
# Runner-side measurement is fixed to: 2 local warmup repeats + 3 timed repeats (median).
frames="${3:-0}"; warmup="${4:-20}"; device=cuda:0; log_dir=logs
run_case() {
  local module="$1" data_root="$2" split="$3" sweeps="$4" run_dtype="$dtype"
  [[ "$backend" == "minkowski" ]] && run_dtype=fp32
  python3 -m "$module" --backend "$backend" --dtype "$run_dtype" --data-root "$data_root" --split "$split" --frames "$frames" --warmup "$warmup" --timing-repeats 3 --sweeps "$sweeps" --device "$device" --log-dir "$log_dir"
}
run_case gtsparse.e2e_v2.kitti_second dataset/kitti test 1
run_case gtsparse.e2e_v2.nuscenes_voxelnext dataset/nuscenes test 1
run_case gtsparse.e2e_v2.nuscenes_voxelnext dataset/nuscenes test 10
run_case gtsparse.e2e_v2.semantickitti_sparse_resunet42 dataset/semantickitti val 1
