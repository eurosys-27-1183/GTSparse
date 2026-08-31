#!/usr/bin/env python3

import torch
import torch.nn.functional as F

from gtsparse.sparse3d.geometric_template import (
    PeriodicTemplateSparseConv3d,
    PeriodicTemplateSparseInverseConv3d,
    PeriodicTemplateSubMConv3d,
)
from gtsparse.sparse3d.reference_ops import (
    compare_sparse_outputs,
    gather_dense_features_at_coords,
    reference_sparse_conv3d,
    reference_subm_conv3d,
)
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor


def make_tensor(shape, channels, dtype, device):
    coords = (torch.rand((1, *shape), device=device) < 0.18).nonzero().int()
    features = torch.randn((coords.size(0), channels), device=device, dtype=dtype)
    return GTSparseSparseConvTensor(features, coords, shape, 1)


def dense_weight(module, transposed=False):
    if transposed:
        return module.weight.permute(1, 2, 0).reshape(
            module.in_channels,
            module.out_channels,
            *module.kernel_size,
        )
    return module.weight.permute(2, 1, 0).reshape(
        module.out_channels,
        module.in_channels,
        *module.kernel_size,
    )


def validate_subm(dtype, device, atol, rtol):
    for kernel, padding in (
        ((3, 3, 1), (1, 1, 0)),
        ((5, 5, 5), (2, 2, 2)),
    ):
        x = make_tensor((8, 8, 8), 6, dtype, device)
        module = PeriodicTemplateSubMConv3d(
            6,
            7,
            kernel,
            padding=padding,
            bias=True,
        ).to(device=device, dtype=dtype).eval()
        actual = module(x)
        expected = reference_subm_conv3d(
            x,
            dense_weight(module),
            module.bias,
            padding=padding,
        )
        ok, message = compare_sparse_outputs(actual, expected, atol=atol, rtol=rtol)
        if not ok:
            raise AssertionError(f"{dtype} SubM {kernel}: {message}")
        error = float((actual.features - expected.features).abs().max())
        print(f"{dtype} SubM {kernel}: max error {error:.8f}")


def validate_full(dtype, device, atol, rtol):
    cases = (
        ((2, 2, 2), (2, 2, 2), (0, 0, 0), (8, 8, 8)),
        ((3, 1, 1), (2, 1, 1), (0, 0, 0), (9, 6, 6)),
    )
    for kernel, stride, padding, shape in cases:
        x = make_tensor(shape, 6, dtype, device)
        module = PeriodicTemplateSparseConv3d(
            6,
            7,
            kernel,
            stride=stride,
            padding=padding,
            bias=True,
        ).to(device=device, dtype=dtype).eval()
        actual = module(x)
        expected = reference_sparse_conv3d(
            x,
            dense_weight(module),
            module.bias,
            stride=stride,
            padding=padding,
        )
        ok, message = compare_sparse_outputs(actual, expected, atol=atol, rtol=rtol)
        if not ok:
            raise AssertionError(f"{dtype} full {kernel}: {message}")
        print(f"{dtype} full {kernel}: {actual.features.size(0)} output rows")


def validate_inverse(dtype, device, atol, rtol):
    x = make_tensor((8, 8, 8), 6, dtype, device)
    down = PeriodicTemplateSparseConv3d(
        6,
        7,
        2,
        stride=2,
        build_reverse=True,
    ).to(device=device, dtype=dtype).eval()
    up = PeriodicTemplateSparseInverseConv3d(7, 5, 2, bias=True).to(
        device=device,
        dtype=dtype,
    ).eval()
    y = down(x)
    actual = up(y)
    expected_dense = F.conv_transpose3d(
        y.dense(),
        dense_weight(up, transposed=True),
        up.bias,
        stride=2,
    )
    expected = gather_dense_features_at_coords(expected_dense, x.indices)
    if not torch.equal(actual.indices, x.indices):
        raise AssertionError(f"{dtype} inverse coordinates differ")
    if not torch.allclose(actual.features, expected, atol=atol, rtol=rtol):
        error = float((actual.features - expected).abs().max())
        raise AssertionError(f"{dtype} inverse max error {error}")
    runtime = y.metadata.reverse_chain[0].runtime
    counts = [int(value) for value in runtime.template_counts.cpu()]
    if sum(counts[8:]) != 0:
        raise AssertionError(f"{dtype} inverse used non-singleton templates: {counts}")
    error = float((actual.features - expected).abs().max())
    print(f"{dtype} inverse 2x2x2: max error {error:.8f}, singleton counts {counts[:8]}")


def main():
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    for dtype, atol, rtol in (
        (torch.float32, 1e-3, 1e-3),
        (torch.float16, 5e-3, 5e-3),
    ):
        validate_subm(dtype, device, atol, rtol)
        validate_full(dtype, device, atol, rtol)
        validate_inverse(dtype, device, atol, rtol)


if __name__ == "__main__":
    main()
