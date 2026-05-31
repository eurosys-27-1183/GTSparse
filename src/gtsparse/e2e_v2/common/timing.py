from __future__ import annotations

import torch


def measure_cuda_elapsed_ms(fn, *args, device: torch.device, repeats: int = 1, warmup_repeats: int = 0, **kwargs):
    repeat_count = max(1, int(repeats))
    warmup_count = max(0, int(warmup_repeats))
    out = None
    for _ in range(warmup_count):
        out = fn(*args, **kwargs)
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(repeat_count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        event_pairs.append((start, end))
    torch.cuda.synchronize(device=device)
    times = sorted(float(start.elapsed_time(end)) for start, end in event_pairs)
    return out, times[len(times) // 2]
