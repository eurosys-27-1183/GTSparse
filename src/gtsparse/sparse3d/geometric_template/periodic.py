from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Tuple

import torch
import torch.nn as nn

from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

from .metadata import GeometricTemplateMetadata, GeometricTemplateReverseEdge
from .tensor import tensor_metadata


def _normalize_3tuple(value) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    values = tuple(int(v) for v in value)
    if len(values) != 3:
        raise ValueError("expected an int or length-3 sequence")
    return values


def _kernel_offsets(kernel_size: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    offsets = tuple(product(*(range(size) for size in kernel_size)))
    return torch.tensor(offsets, device=device, dtype=torch.int64)


def _template_slots(kernel_size: Tuple[int, int, int], period: int) -> tuple[tuple[int, ...], ...]:
    offsets = tuple(product(*(range(size) for size in kernel_size)))
    volume = len(offsets)
    candidates: list[tuple[int, ...]] = []

    if volume <= 8:
        candidates.extend((slot,) for slot in range(volume))

    center = None
    if all(size % 2 == 1 for size in kernel_size):
        center_coord = tuple(size // 2 for size in kernel_size)
        center = offsets.index(center_coord)
        candidates.insert(0, (center,))

    axis = max(range(3), key=lambda dim: kernel_size[dim])
    phase_period = min(int(period), kernel_size[axis])
    for phase in range(phase_period):
        keep = [slot for slot, offset in enumerate(offsets) if offset[axis] % phase_period == phase]
        if center is not None and center not in keep:
            keep.append(center)
        candidates.append(tuple(sorted(keep)))
    for phase in range(phase_period):
        keep = [slot for slot, offset in enumerate(offsets) if offset[axis] % phase_period != phase]
        if center is not None and center not in keep:
            keep.append(center)
        candidates.append(tuple(sorted(keep)))
    candidates.append(tuple(range(volume)))

    unique = []
    seen = set()
    for slots in sorted(candidates, key=len):
        if slots and slots not in seen:
            unique.append(slots)
            seen.add(slots)
    return tuple(unique)


def _linear_keys(coords: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
    d, h, w = spatial_shape
    coords = coords.to(torch.int64)
    return ((coords[:, 0] * d + coords[:, 1]) * h + coords[:, 2]) * w + coords[:, 3]


def _input_rows(
    in_coords: torch.Tensor,
    out_coords: torch.Tensor,
    in_spatial: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> torch.Tensor:
    n_out = int(out_coords.size(0))
    volume = math.prod(kernel_size)
    if n_out == 0 or int(in_coords.size(0)) == 0:
        return torch.full((n_out, volume), -1, device=out_coords.device, dtype=torch.int64)

    offsets = _kernel_offsets(kernel_size, out_coords.device)
    out_xyz = out_coords[:, None, 1:].to(torch.int64)
    stride_t = torch.tensor(stride, device=out_coords.device, dtype=torch.int64)
    padding_t = torch.tensor(padding, device=out_coords.device, dtype=torch.int64)
    dilation_t = torch.tensor(dilation, device=out_coords.device, dtype=torch.int64)
    query_xyz = out_xyz * stride_t - padding_t + offsets[None] * dilation_t
    spatial_t = torch.tensor(in_spatial, device=out_coords.device, dtype=torch.int64)
    valid = ((query_xyz >= 0) & (query_xyz < spatial_t)).all(dim=2)
    query_batch = out_coords[:, None, 0].to(torch.int64).expand(-1, volume)
    query_coords = torch.cat((query_batch[..., None], query_xyz), dim=2).reshape(-1, 4)
    query_keys = _linear_keys(query_coords, in_spatial)

    input_keys = _linear_keys(in_coords, in_spatial)
    sorted_keys, order = torch.sort(input_keys)
    positions = torch.searchsorted(sorted_keys, query_keys)
    safe_positions = positions.clamp_max(int(sorted_keys.numel()) - 1)
    matched = valid.reshape(-1) & positions.lt(sorted_keys.numel())
    matched &= sorted_keys.index_select(0, safe_positions) == query_keys
    rows = torch.full_like(query_keys, -1)
    rows[matched] = order.index_select(0, safe_positions[matched])
    return rows.view(n_out, volume)


def _output_coords(
    in_coords: torch.Tensor,
    out_spatial: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> torch.Tensor:
    volume = math.prod(kernel_size)
    if int(in_coords.size(0)) == 0:
        return torch.empty((0, 4), device=in_coords.device, dtype=torch.int32)

    offsets = _kernel_offsets(kernel_size, in_coords.device)
    in_xyz = in_coords[:, None, 1:].to(torch.int64)
    stride_t = torch.tensor(stride, device=in_coords.device, dtype=torch.int64)
    padding_t = torch.tensor(padding, device=in_coords.device, dtype=torch.int64)
    dilation_t = torch.tensor(dilation, device=in_coords.device, dtype=torch.int64)
    numer = in_xyz + padding_t - offsets[None] * dilation_t
    divisible = torch.remainder(numer, stride_t).eq(0).all(dim=2)
    out_xyz = torch.div(numer, stride_t, rounding_mode="floor")
    spatial_t = torch.tensor(out_spatial, device=in_coords.device, dtype=torch.int64)
    valid = divisible & ((out_xyz >= 0) & (out_xyz < spatial_t)).all(dim=2)
    batches = in_coords[:, None, 0].to(torch.int64).expand(-1, volume)
    candidates = torch.cat((batches[..., None], out_xyz), dim=2)[valid]
    return torch.unique(candidates, dim=0, sorted=True).to(torch.int32)


@dataclass(slots=True)
class PeriodicTemplateRuntime:
    kernel_size: Tuple[int, int, int]
    input_rows: torch.Tensor
    row_groups: tuple[torch.Tensor, ...]
    template_slots: tuple[torch.Tensor, ...]
    template_ids: torch.Tensor
    template_counts: torch.Tensor
    out_coords: torch.Tensor
    out_spatial: Tuple[int, int, int]

    @property
    def n_out(self) -> int:
        return int(self.out_coords.size(0))


def _make_runtime(
    input_rows: torch.Tensor,
    out_coords: torch.Tensor,
    out_spatial: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
    period: int,
) -> PeriodicTemplateRuntime:
    slot_tuples = _template_slots(kernel_size, period)
    template_slots = tuple(
        torch.tensor(slots, device=input_rows.device, dtype=torch.int64)
        for slots in slot_tuples
    )
    active = input_rows.ge(0)
    active_count = active.sum(dim=1)
    unassigned = torch.ones((input_rows.size(0),), device=input_rows.device, dtype=torch.bool)
    template_ids = torch.full((input_rows.size(0),), -1, device=input_rows.device, dtype=torch.int32)
    row_groups = []
    for template_id, slots in enumerate(template_slots):
        covered = active.index_select(1, slots).sum(dim=1)
        rows = torch.nonzero(unassigned & covered.eq(active_count), as_tuple=False).flatten()
        template_ids[rows] = template_id
        unassigned[rows] = False
        row_groups.append(rows)
    template_counts = torch.tensor(
        [rows.numel() for rows in row_groups],
        device=input_rows.device,
        dtype=torch.int32,
    )
    return PeriodicTemplateRuntime(
        kernel_size=kernel_size,
        input_rows=input_rows,
        row_groups=tuple(row_groups),
        template_slots=template_slots,
        template_ids=template_ids,
        template_counts=template_counts,
        out_coords=out_coords,
        out_spatial=out_spatial,
    )


def build_periodic_subm_runtime(
    x: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    period: int,
) -> PeriodicTemplateRuntime:
    key = f"periodic:{x.indices.data_ptr()}:{kernel_size}:{padding}:{dilation}:{period}"
    runtime = x._runtime_cache.get(key)
    if runtime is None:
        spatial = tuple(int(v) for v in x.spatial_shape)
        rows = _input_rows(x.indices, x.indices, spatial, kernel_size, (1, 1, 1), padding, dilation)
        runtime = _make_runtime(rows, x.indices, spatial, kernel_size, period)
        x._runtime_cache[key] = runtime
    return runtime


def build_periodic_full_runtime(
    x: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    period: int,
) -> PeriodicTemplateRuntime:
    in_spatial = tuple(int(v) for v in x.spatial_shape)
    out_spatial = tuple(
        (in_spatial[dim] + 2 * padding[dim] - dilation[dim] * (kernel_size[dim] - 1) - 1)
        // stride[dim]
        + 1
        for dim in range(3)
    )
    out_coords = _output_coords(x.indices, out_spatial, kernel_size, stride, padding, dilation)
    rows = _input_rows(x.indices, out_coords, in_spatial, kernel_size, stride, padding, dilation)
    return _make_runtime(rows, out_coords, out_spatial, kernel_size, period)


def build_periodic_reverse_runtime(
    forward: PeriodicTemplateRuntime,
    out_coords: torch.Tensor,
    out_spatial: Tuple[int, int, int],
    period: int,
) -> PeriodicTemplateRuntime:
    volume = math.prod(forward.kernel_size)
    reverse_rows = torch.full(
        (out_coords.size(0), volume),
        -1,
        device=forward.input_rows.device,
        dtype=torch.int64,
    )
    forward_rows = torch.arange(forward.n_out, device=forward.input_rows.device, dtype=torch.int64)
    forward_rows = forward_rows[:, None].expand(-1, volume)
    slots = torch.arange(volume, device=forward.input_rows.device, dtype=torch.int64)
    slots = slots[None].expand(forward.n_out, -1)
    valid = forward.input_rows.ge(0)
    reverse_rows[forward.input_rows[valid], slots[valid]] = forward_rows[valid]
    return _make_runtime(reverse_rows, out_coords, out_spatial, forward.kernel_size, period)


def periodic_template_conv(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: PeriodicTemplateRuntime,
) -> torch.Tensor:
    volume = math.prod(runtime.kernel_size)
    if weight.dim() != 3 or int(weight.size(0)) != volume:
        raise ValueError(f"weight must have shape [{volume}, Cin, Cout]")
    if features.dtype not in (torch.float16, torch.float32):
        raise TypeError("periodic template convolution requires float16 or float32 features")

    cin = int(features.size(1))
    cout = int(weight.size(2))
    padded_features = torch.cat((features, features.new_zeros((1, cin))), dim=0)
    output = features.new_zeros((runtime.n_out, cout))
    zero_row = int(features.size(0))
    for rows, slots in zip(runtime.row_groups, runtime.template_slots):
        if rows.numel() == 0:
            continue
        input_rows = runtime.input_rows.index_select(0, rows).index_select(1, slots)
        input_rows = torch.where(input_rows >= 0, input_rows, zero_row)
        gathered = padded_features.index_select(0, input_rows.flatten())
        gathered = gathered.view(rows.numel(), slots.numel() * cin)
        packed_weight = weight.index_select(0, slots).reshape(slots.numel() * cin, cout)
        output.index_copy_(0, rows, gathered @ packed_weight)
    return output


class PeriodicTemplateSubMConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        *,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = False,
        period: int = 3,
    ) -> None:
        super().__init__()
        if int(groups) != 1:
            raise ValueError("periodic template convolution requires groups=1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _normalize_3tuple(kernel_size)
        self.padding = _normalize_3tuple(padding)
        self.dilation = _normalize_3tuple(dilation)
        self.period = int(period)
        volume = math.prod(self.kernel_size)
        self.weight = nn.Parameter(torch.empty(volume, self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(float(self.in_channels * math.prod(self.kernel_size)))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        runtime = build_periodic_subm_runtime(
            x,
            self.kernel_size,
            self.padding,
            self.dilation,
            self.period,
        )
        features = periodic_template_conv(x.features, self.weight, runtime)
        if self.bias is not None:
            features = features + self.bias
        return x.replace_feature(features)


class PeriodicTemplateSparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        *,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = False,
        build_reverse: bool = True,
        period: int = 3,
    ) -> None:
        super().__init__()
        if int(groups) != 1:
            raise ValueError("periodic template convolution requires groups=1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _normalize_3tuple(kernel_size)
        self.stride = _normalize_3tuple(stride)
        self.padding = _normalize_3tuple(padding)
        self.dilation = _normalize_3tuple(dilation)
        self.build_reverse = bool(build_reverse)
        self.period = int(period)
        volume = math.prod(self.kernel_size)
        self.weight = nn.Parameter(torch.empty(volume, self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(float(self.in_channels * math.prod(self.kernel_size)))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        metadata = tensor_metadata(x)
        runtime = build_periodic_full_runtime(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.period,
        )
        features = periodic_template_conv(x.features, self.weight, runtime)
        if self.bias is not None:
            features = features + self.bias
        reverse_chain = metadata.reverse_chain
        if self.build_reverse:
            reverse = build_periodic_reverse_runtime(
                runtime,
                x.indices,
                tuple(int(v) for v in x.spatial_shape),
                self.period,
            )
            reverse_chain = (GeometricTemplateReverseEdge(reverse, x.coord_hashmap), *reverse_chain)
        out_metadata = GeometricTemplateMetadata(subm_runtime=None, reverse_chain=reverse_chain)
        return x.replace_sparse(
            new_features=features,
            new_coords=runtime.out_coords,
            new_spatial_shape=runtime.out_spatial,
            new_batch_size=x.batch_size,
            coord_hashmap=None,
            metadata=out_metadata,
        )


class PeriodicTemplateSparseInverseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        *,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if int(groups) != 1:
            raise ValueError("periodic template convolution requires groups=1")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _normalize_3tuple(kernel_size)
        volume = math.prod(self.kernel_size)
        self.weight = nn.Parameter(torch.empty(volume, self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(float(self.in_channels * math.prod(self.kernel_size)))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        metadata = tensor_metadata(x)
        if not metadata.reverse_chain:
            raise RuntimeError("inverse conv requires reverse metadata")
        edge = metadata.reverse_chain[0]
        runtime = edge.runtime
        if not isinstance(runtime, PeriodicTemplateRuntime) or runtime.kernel_size != self.kernel_size:
            raise RuntimeError("inverse conv does not match the preceding periodic full convolution")
        features = periodic_template_conv(x.features, self.weight, runtime)
        if self.bias is not None:
            features = features + self.bias
        out_metadata = GeometricTemplateMetadata(
            subm_runtime=None,
            reverse_chain=metadata.reverse_chain[1:],
        )
        return x.replace_sparse(
            new_features=features,
            new_coords=runtime.out_coords,
            new_spatial_shape=runtime.out_spatial,
            new_batch_size=x.batch_size,
            coord_hashmap=edge.coord_hashmap,
            metadata=out_metadata,
        )
