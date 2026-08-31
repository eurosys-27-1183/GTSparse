import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.workloads import WORKLOADS, build_workload, move_batch, sparse_forward
from gtsparse.sparse3d.geometric_template.runtime import PAYLOAD_LOGICAL_TO_ACTUAL, TEMPLATE_KEEP_SLOTS
import gtsparse.sparse3d.geometric_template.ops as gt_ops


capture = False
current_layers = []
original_conv = gt_ops._conv


def _template_buffer(runtime, template_id: int, count: int):
    if template_id == 0:
        return runtime.input_rows_w1[0, :count, :1]
    if template_id < 4:
        return runtime.input_rows_w9[template_id - 1, :count, : len(TEMPLATE_KEEP_SLOTS[template_id])]
    if template_id < 7:
        return runtime.input_rows_w18[template_id - 4, :count, : len(TEMPLATE_KEEP_SLOTS[template_id])]
    return runtime.input_rows_w27[0, :count, :27]


def _runtime_masks(runtime, counts: list[int]) -> np.ndarray:
    all_masks = []
    for template_id, count in enumerate(counts):
        if count == 0:
            continue
        buffer = _template_buffer(runtime, template_id, count)
        actual_offsets = [PAYLOAD_LOGICAL_TO_ACTUAL[slot] for slot in TEMPLATE_KEEP_SLOTS[template_id]]
        bits = torch.tensor([1 << offset for offset in actual_offsets], device=buffer.device, dtype=torch.int64)
        masks = ((buffer >= 0).to(torch.int64) * bits).sum(dim=1).cpu().numpy().astype(np.uint32)
        all_masks.append(masks)
    return np.concatenate(all_masks) if all_masks else np.empty(0, dtype=np.uint32)


def _spconv_offset_rows(masks: np.ndarray, tile_rows: int) -> int:
    masks = np.sort(masks)
    issued = 0
    for start in range(0, len(masks), tile_rows):
        tile = masks[start : start + tile_rows]
        issued += len(tile) * int(int(np.bitwise_or.reduce(tile)).bit_count())
    return issued


def observe_runtime(kind, features, logical_weight, runtime) -> None:
    counts = [int(value) for value in runtime.template_counts.tolist()]
    padded_counts = [int(value) for value in runtime.padded_counts.tolist()]
    masks = _runtime_masks(runtime, counts)
    active_pairs = int(sum(int(value).bit_count() for value in masks))
    widths = (1, 10, 10, 10, 19, 19, 19, 27)
    gtsparse_offset_rows = sum(count * width for count, width in zip(counts, widths))
    full_offset_rows = int(runtime.n_out) * 27
    spconv_offset_rows = {tile_rows: _spconv_offset_rows(masks, tile_rows) for tile_rows in (32, 64, 128)}
    cin = int(features.size(1))
    cout = int(logical_weight.size(-1))
    flops_per_pair = 2 * cin * cout
    current_layers.append(
        {
            "active_pairs": active_pairs,
            "cin": cin,
            "cout": cout,
            "effective_flops": active_pairs * flops_per_pair,
            "gtsparse_issued_flops": gtsparse_offset_rows * flops_per_pair,
            "kind": kind,
            "minkowski_issued_flops": active_pairs * flops_per_pair,
            "n_out": int(runtime.n_out),
            "padded_counts": padded_counts,
            "spconv_issued_flops_bm32": spconv_offset_rows[32] * flops_per_pair,
            "spconv_issued_flops_bm64": spconv_offset_rows[64] * flops_per_pair,
            "spconv_issued_flops_bm128": spconv_offset_rows[128] * flops_per_pair,
            "template_counts": counts,
            "torchsparse_issued_flops": full_offset_rows * flops_per_pair,
        }
    )


def observed_conv(features, logical_weight, runtime):
    if capture:
        observe_runtime("native", features, logical_weight, runtime)
    return original_conv(features, logical_weight, runtime)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    global capture
    args = parse_args()
    gt_ops._conv = observed_conv
    model, loader, dtype = build_workload(args.workload, "gtsparse", "fp16", args.frames, args.device)

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.warmup:
                break
            sparse_forward(model, move_batch(batch, args.device, dtype), args.workload)
    torch.cuda.synchronize(args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    capture = True
    with args.out.open("w", encoding="utf-8") as output:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"template-profile/{args.workload}", dynamic_ncols=True):
                current_layers.clear()
                batch = move_batch(batch, args.device, dtype)
                sparse_forward(model, batch, args.workload)
                torch.cuda.synchronize(args.device)
                family_counts = [0, 0, 0, 0]
                for layer in current_layers:
                    counts = layer["template_counts"]
                    family_counts[0] += counts[0]
                    family_counts[1] += sum(counts[1:4])
                    family_counts[2] += sum(counts[4:7])
                    family_counts[3] += counts[7]
                record = {
                    "effective_flops": sum(layer["effective_flops"] for layer in current_layers),
                    "family_counts": family_counts,
                    "frame_ids": list(batch.frame_ids),
                    "gpu": torch.cuda.get_device_name(torch.device(args.device)),
                    "gtsparse_issued_flops": sum(layer["gtsparse_issued_flops"] for layer in current_layers),
                    "layers": current_layers,
                    "minkowski_issued_flops": sum(layer["minkowski_issued_flops"] for layer in current_layers),
                    "torchsparse_issued_flops": sum(layer["torchsparse_issued_flops"] for layer in current_layers),
                    "workload": args.workload,
                }
                json.dump(record, output, sort_keys=True)
                output.write("\n")
                output.flush()


if __name__ == "__main__":
    main()
