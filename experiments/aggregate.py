import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(root: Path):
    rows = []
    per_frame = []
    for path in sorted(root.glob("**/*.summary.json")):
        data = json.loads(path.read_text())
        if "stats" not in data or "workload" not in data:
            continue
        relative = path.relative_to(root)
        experiment = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        min_template_match = re.search(r"min_template_(\d+)", str(relative))
        for metric, stats in data["stats"].items():
            if stats is None:
                continue
            rows.append(
                {
                    "backend": data["backend"],
                    "count": stats["count"],
                    "dtype": data["dtype"],
                    "experiment": experiment,
                    "gpu": data["gpu"],
                    "mean_ms": stats["mean_ms"],
                    "median_ms": stats["median_ms"],
                    "metric": metric,
                    "min_template": data.get("min_template", "" if min_template_match is None else min_template_match.group(1)),
                    "sweeps": data["sweeps"],
                    "workload": data["workload"],
                }
            )
        raw_path = path.with_name(path.name.replace(".summary.json", ".jsonl"))
        if experiment == "end_to_end" and raw_path.exists():
            for frame_index, record in enumerate(read_jsonl(raw_path)):
                base = {
                    "backend": data["backend"],
                    "dtype": data["dtype"],
                    "frame_id": record["frame_ids"][0],
                    "frame_index": frame_index,
                    "gpu": data["gpu"],
                    "workload": data["workload"],
                }
                if "conv_only_ms" in record:
                    per_frame.append({**base, "latency_ms": record["conv_only_ms"], "metric": "conv_only"})
                if "end2end_ms" in record:
                    per_frame.append({**base, "latency_ms": record["end2end_ms"], "metric": "end2end"})
    return rows, per_frame


def profile_rows(root: Path):
    profiles = []
    raw_by_key = {}
    for path in sorted((root / "microbenchmark" / "profile").glob("*.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        raw_by_key[(records[0]["gpu"], records[0]["workload"])] = records
    for path in sorted((root / "microbenchmark" / "template_profile").glob("*.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        counts = [sum(record["family_counts"][index] for record in records) for index in range(4)]
        total = sum(counts)
        if total == 0:
            raise ValueError(f"template profile contains no K=27 rows: {path}")
        widths = (1, 10, 19, 27)
        row = {
            "avg_assigned_width": sum(count * width for count, width in zip(counts, widths)) / total,
            "center_percent": 100 * counts[0] / total,
            "frames": len(records),
            "full27_percent": 100 * counts[3] / total,
            "gpu": records[0]["gpu"],
            "operator": "3x3x3",
            "skip1_percent": 100 * counts[2] / total,
            "skip2_percent": 100 * counts[1] / total,
            "workload": records[0]["workload"],
        }
        profiles.append(row)
    spconv_by_key = {}
    for path in sorted((root / "microbenchmark" / "profile_spconv").glob("*.jsonl")):
        records = read_jsonl(path)
        if records:
            spconv_by_key[(records[0]["gpu"], records[0]["workload"])] = records
    return profiles, raw_by_key, spconv_by_key


def timing_rows(root: Path):
    """Return per-frame conv-only timings keyed by backend and workload.

    Throughput is an aggregate over the same frames as the FLOP profile.  The
    timing summary's median is useful for latency tables, but pairing each
    profile record with its frame timing lets the throughput numerator and
    denominator use one consistent sample and avoids mixing mean and median
    statistics.
    """
    timings = {}
    timing_root = root / "microbenchmark" / "timing"
    for summary_path in sorted(timing_root.glob("logs_*/*.summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        path = summary_path.with_name(summary_path.name.replace(".summary.json", ".jsonl"))
        if not path.exists():
            continue
        records = read_jsonl(path)
        if not records:
            continue
        key = (summary["gpu"], summary["workload"], summary["backend"], summary["dtype"])
        timings[key] = {
            tuple(record["frame_ids"]): float(record["conv_only_ms"])
            for record in records
            if "conv_only_ms" in record and record.get("frame_ids")
        }
    return timings


def spconv_issued_flops(profile_record, spconv_record) -> int:
    issued = 0
    if len(profile_record["layers"]) != len(spconv_record["layers"]):
        raise ValueError("GTSparse and SpConv profiles contain different sparse-convolution layer counts")
    for gtsparse_layer, spconv_layer in zip(profile_record["layers"], spconv_record["layers"]):
        for field in ("cin", "cout", "n_out"):
            if gtsparse_layer[field] != spconv_layer[field]:
                raise ValueError(f"GTSparse and SpConv layer profiles differ at {field}")
        issued += gtsparse_layer[f"spconv_issued_flops_bm{spconv_layer['tile_rows']}"]
    return issued


def torchsparse_issued_flops(profile_record, spconv_record) -> int:
    if len(profile_record["layers"]) != len(spconv_record["layers"]):
        raise ValueError("GTSparse and SpConv profiles contain different sparse-convolution layer counts")
    issued = 0
    voxelnext_bev_conv = False
    for gtsparse_layer, spconv_layer in zip(profile_record["layers"], spconv_record["layers"]):
        if profile_record["workload"].startswith("voxelnext_") and gtsparse_layer["kind"] == "kernel9" and not voxelnext_bev_conv:
            issued += gtsparse_layer[f"spconv_issued_flops_bm{spconv_layer['tile_rows']}"]
            voxelnext_bev_conv = True
        else:
            issued += gtsparse_layer["torchsparse_issued_flops"]
    return issued


def throughput_rows(summary, raw_profiles, spconv_profiles, frame_timings=None):
    rows = []
    issued_key = {
        "gtsparse": "gtsparse_issued_flops",
        "minkowski": "minkowski_issued_flops",
    }
    for timing in summary:
        if timing["experiment"] != "microbenchmark" or timing["metric"] != "conv_only":
            continue
        key = (timing["gpu"], timing["workload"])
        records = raw_profiles.get(key)
        if not records:
            continue
        backend = timing["backend"]
        timing_by_frame = None
        if frame_timings is not None:
            timing_by_frame = frame_timings.get((timing["gpu"], timing["workload"], backend, timing["dtype"]))
            if not timing_by_frame:
                continue
            missing = [tuple(record["frame_ids"]) for record in records if tuple(record["frame_ids"]) not in timing_by_frame]
            if missing:
                raise ValueError(
                    f"throughput profile/timing frame mismatch for {timing['gpu']} "
                    f"{timing['workload']} {backend}: {len(missing)} profile frames have no timing"
                )
            elapsed_ms = sum(timing_by_frame[tuple(record["frame_ids"])] for record in records)
            effective = sum(record["effective_flops"] for record in records)
        else:
            # Backward-compatible path for callers that only have summaries.
            elapsed_ms = float(timing["median_ms"]) * len(records)
            effective = statistics.mean(record["effective_flops"] for record in records) * len(records)
        if backend == "spconv":
            spconv_records = spconv_profiles.get(key)
            if not spconv_records:
                continue
            by_frame = {tuple(record["frame_ids"]): record for record in spconv_records}
            issued_values = [spconv_issued_flops(record, by_frame[tuple(record["frame_ids"])]) for record in records]
        elif backend == "torchsparse":
            spconv_records = spconv_profiles.get(key)
            if not spconv_records:
                continue
            by_frame = {tuple(record["frame_ids"]): record for record in spconv_records}
            issued_values = [torchsparse_issued_flops(record, by_frame[tuple(record["frame_ids"])]) for record in records]
        else:
            issued_values = [record[issued_key[backend]] for record in records]
        issued = sum(issued_values)
        rows.append(
            {
                "backend": backend,
                "dtype": timing["dtype"],
                "effective_tflops": effective / elapsed_ms / 1e9,
                "gpu": timing["gpu"],
                "proportionality_percent": 100 * effective / issued,
                "raw_tflops": issued / elapsed_ms / 1e9,
                "workload": timing["workload"],
            }
        )
    return rows


def breakdown_rows(root: Path, summaries):
    rows = []
    for path in sorted((root / "microbenchmark" / "time_breakdown").glob("*.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        backend = records[0]["backend"]
        dtype = records[0]["dtype"]
        gpu = records[0]["gpu"]
        workload = records[0]["workload"]
        metric = "end2end" if workload.startswith("minkunet") else "conv_only"
        targets = [
            row
            for row in summaries
            if row["experiment"] == "end_to_end"
            and row["gpu"] == gpu
            and row["workload"] == workload
            and row["backend"] == backend
            and row["dtype"] == dtype
            and row["metric"] == metric
        ]
        scale_experiment = "end_to_end"
        if not targets:
            targets = [
                row
                for row in summaries
                if row["experiment"] == "microbenchmark"
                and row["gpu"] == gpu
                and row["workload"] == workload
                and row["backend"] == backend
                and row["dtype"] == dtype
                and row["metric"] == metric
            ]
            scale_experiment = "microbenchmark"
        if not targets:
            raise ValueError(f"missing latency scale target for {gpu} {workload} {backend} {dtype}")
        raw_builder = statistics.median(record["builder_ms"] for record in records)
        raw_kernel = statistics.median(record["kernel_ms"] for record in records)
        builder_share = statistics.median(
            record["builder_ms"] / record["total_ms"] for record in records
        )
        target_ms = float(targets[-1]["median_ms"])
        rows.append(
            {
                "backend": backend,
                "builder_median_ms": builder_share * target_ms,
                "builder_percent": 100 * builder_share,
                "dtype": dtype,
                "frames": len(records),
                "gpu": gpu,
                "kernel_median_ms": (1 - builder_share) * target_ms,
                "kernel_percent": 100 * (1 - builder_share),
                "method": records[0]["method"],
                "raw_builder_median_ms": raw_builder,
                "raw_kernel_median_ms": raw_kernel,
                "scale_experiment": scale_experiment,
                "scale_metric": metric,
                "total_median_ms": target_ms,
                "workload": workload,
            }
        )
    return rows


def memory_rows(root: Path):
    rows = []
    for path in sorted((root / "microbenchmark" / "peak_memory").glob("*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries, per_frame = summary_rows(args.logs)
    profiles, raw_profiles, spconv_profiles = profile_rows(args.logs)
    frame_timings = timing_rows(args.logs)
    throughput = throughput_rows(summaries, raw_profiles, spconv_profiles, frame_timings)
    breakdown = breakdown_rows(args.logs, summaries)
    memory = memory_rows(args.logs)

    write_csv(args.results / "latency.csv", ("experiment", "gpu", "dtype", "workload", "sweeps", "backend", "metric", "count", "median_ms", "mean_ms", "min_template"), summaries)
    write_csv(args.results / "per_frame_latency.csv", ("gpu", "dtype", "workload", "backend", "metric", "frame_index", "frame_id", "latency_ms"), per_frame)
    write_csv(args.results / "template_distribution.csv", ("gpu", "workload", "operator", "frames", "center_percent", "skip2_percent", "skip1_percent", "full27_percent", "avg_assigned_width"), profiles)
    write_csv(args.results / "effective_throughput.csv", ("gpu", "workload", "backend", "dtype", "raw_tflops", "effective_tflops", "proportionality_percent"), throughput)
    write_csv(
        args.results / "time_breakdown.csv",
        (
            "gpu",
            "workload",
            "backend",
            "dtype",
            "frames",
            "method",
            "scale_experiment",
            "scale_metric",
            "builder_percent",
            "kernel_percent",
            "raw_builder_median_ms",
            "raw_kernel_median_ms",
            "builder_median_ms",
            "kernel_median_ms",
            "total_median_ms",
        ),
        breakdown,
    )
    write_csv(args.results / "peak_memory.csv", ("gpu", "workload", "backend", "dtype", "frames", "peak_memory_mb"), memory)


if __name__ == "__main__":
    main()
