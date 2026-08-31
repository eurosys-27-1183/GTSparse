#!/usr/bin/env python3

import argparse

import torch

from gtsparse.sparse3d.geometric_template import GeometricTemplateKernel3Conv3d
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def linear_keys(coords, spatial):
    depth, height, width = (int(value) for value in spatial)
    coords = coords.to(torch.int64)
    return ((coords[:, 0] * depth + coords[:, 1]) * height + coords[:, 2]) * width + coords[:, 3]


def reference(features, weight, input_coords, output_coords, input_spatial):
    sorted_keys, order = torch.sort(linear_keys(input_coords, input_spatial))
    output = torch.zeros(
        (output_coords.size(0), weight.size(2)),
        device=features.device,
        dtype=features.dtype,
    )
    for offset in range(3):
        query_coords = output_coords.clone()
        query_coords[:, 1] = query_coords[:, 1] * 2 + offset
        query_keys = linear_keys(query_coords, input_spatial)
        positions = torch.searchsorted(sorted_keys, query_keys)
        safe = positions.clamp_max(sorted_keys.numel() - 1)
        found = positions.lt(sorted_keys.numel()) & sorted_keys[safe].eq(query_keys)
        input_rows = order[safe[found]]
        output[found] += features[input_rows] @ weight[offset]
    return output


def validate_dtype(dtype, device):
    repeats = 300
    input_spatial = (6, repeats * 7 + 1, 1)
    coords = []
    expected_coords = set()
    for repeat in range(repeats):
        for mask in range(1, 8):
            height = repeat * 7 + mask
            for offset in range(3):
                if mask & (1 << offset):
                    coords.append((0, 2 + offset, height, 0))
            expected_coords.add((0, 1, height, 0))
            if mask & 1:
                expected_coords.add((0, 0, height, 0))

    coords = torch.tensor(sorted(coords), device=device, dtype=torch.int32)
    expected_coords = torch.tensor(sorted(expected_coords), device=device, dtype=torch.int32)
    features = (torch.randn(coords.size(0), 64, device=device) * 0.1).to(dtype)
    module = GeometricTemplateKernel3Conv3d(64, 128, stride=(2, 1, 1)).to(device=device, dtype=dtype).eval()
    sparse_input = GTSparseSparseConvTensor(features, coords, input_spatial, 1)
    with torch.inference_mode():
        output = module(sparse_input)

    runtime, _ = module.build_runtime(sparse_input)
    expected_counts = [repeats, repeats, repeats * 5, repeats, repeats, repeats, repeats]
    assert runtime.template_counts.cpu().tolist() == expected_counts
    assert torch.equal(output.indices, expected_coords)

    padded_rows = int(runtime.padded_counts.sum().item())
    live_out_rows = runtime.out_rows[:padded_rows]
    live_out_rows = live_out_rows[live_out_rows.ge(0)]
    assert live_out_rows.numel() == output.features.size(0)
    assert torch.unique(live_out_rows).numel() == output.features.size(0)
    assert runtime.out_rows[padded_rows:].eq(-1).all()
    assert runtime.template_ids[padded_rows:].eq(-1).all()

    expected_features = reference(
        features,
        module.weight.detach(),
        coords,
        output.indices,
        input_spatial,
    )
    error = (output.features - expected_features).abs().max().item()
    tolerance = 1e-5 if dtype == torch.float32 else 2e-3
    assert error <= tolerance, f"{dtype} max error {error} exceeds {tolerance}"
    average_width = sum(
        count * (1 if template < 3 else 2 if template < 6 else 3)
        for template, count in enumerate(expected_counts)
    ) / sum(expected_counts)
    print(f"{dtype} rows={output.features.size(0)} average_width={average_width:.5f} max_error={error:.9g}")


def main():
    args = parse_args()
    torch.manual_seed(0)
    validate_dtype(torch.float32, args.device)
    validate_dtype(torch.float16, args.device)


if __name__ == "__main__":
    main()
