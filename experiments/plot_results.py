import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


BACKENDS = ("minkowski", "spconv", "torchsparse", "gtsparse")
LABELS = {"minkowski": "Minkowski", "spconv": "SpConv", "torchsparse": "TS++", "gtsparse": "GTSparse"}
COLORS = {"minkowski": "#b3cde3", "spconv": "#ccebc5", "torchsparse": "#fddaec", "gtsparse": "#fed9a6"}
HATCHES = {"minkowski": "//", "spconv": "\\\\", "torchsparse": "xx", "gtsparse": ""}
LINE_COLORS = {"minkowski": "#3a7cb8", "spconv": "#2d8e2d", "torchsparse": "#c4386b", "gtsparse": "#d4760a"}
MARKERS = {"minkowski": "s", "spconv": "^", "torchsparse": "x", "gtsparse": "o"}
WORKLOAD_LABELS = {
    "second_kitti_sweeps1": "SECOND",
    "voxelnext_nuscenes_sweeps1": "VoxelNeXt sw=1",
    "voxelnext_nuscenes_sweeps10": "VoxelNeXt sw=10",
    "minkunet_semantickitti_sweeps1": "MinkUNet42",
}


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def table_canvas(column_widths, row_count: int, header_rows: int, figure_width: float, caption: str):
    total_width = float(sum(column_widths))
    table_height = row_count + header_rows
    fig, axis = plt.subplots(figsize=(figure_width, 0.36 * table_height + 0.7))
    axis.axis("off")
    axis.set_xlim(0.0, total_width)
    axis.set_ylim(0.0, float(table_height) + 1.0)
    axis.text(total_width / 2, table_height + 0.62, caption, ha="center", va="center", fontsize=9.5, linespacing=1.25)
    edges = np.concatenate(([0.0], np.cumsum(column_widths)))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return fig, axis, edges, centers


def table_rule(axis, y: float, start: float, end: float, linewidth: float) -> None:
    axis.plot((start, end), (y, y), color="black", linewidth=linewidth, clip_on=False)


def plot_end_to_end(rows, figures: Path, dtype: str) -> None:
    rows = [row for row in rows if row["experiment"] == "end_to_end"]
    gpus = sorted({row["gpu"] for row in rows if row["backend"] == "gtsparse" and row["dtype"] == dtype})
    workloads = [name for name in WORKLOAD_LABELS if any(row["workload"] == name for row in rows)]
    if not gpus or not workloads:
        return
    fig, axes = plt.subplots(len(gpus), len(workloads), figsize=(2.2 * len(workloads), 1.9 * len(gpus)), squeeze=False)
    for gpu_index, gpu in enumerate(gpus):
        gpu_label = gpu.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        for workload_index, workload in enumerate(workloads):
            axis = axes[gpu_index][workload_index]
            metric = "end2end" if workload.startswith("minkunet") else "conv_only"
            values = []
            for backend in BACKENDS:
                actual_dtype = "fp32" if backend == "minkowski" else dtype
                match = [row for row in rows if row["gpu"] == gpu and row["workload"] == workload and row["backend"] == backend and row["dtype"] == actual_dtype and row["metric"] == metric]
                values.append(float(match[-1]["median_ms"]) if match else np.nan)
            for index, (backend, value) in enumerate(zip(BACKENDS, values)):
                axis.bar(
                    index,
                    value,
                    0.62,
                    color=COLORS[backend],
                    hatch=HATCHES[backend],
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=3,
                )
            axis.set_title(WORKLOAD_LABELS[workload], fontsize=10)
            axis.set_xticks([])
            axis.tick_params(axis="y", labelsize=9, pad=1, length=2)
            axis.set_axisbelow(True)
            axis.grid(axis="y", alpha=0.25, linewidth=0.4)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if workload_index == 0:
                axis.set_ylabel(f"{gpu_label} ({dtype.upper()})\nLatency (ms)", fontsize=9.5)
    handles = [Patch(facecolor=COLORS[name], hatch=HATCHES[name], edgecolor="black", linewidth=0.5, label=LABELS[name]) for name in BACKENDS]
    legend = fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=10, frameon=False)
    for text in legend.get_texts():
        if text.get_text() == "GTSparse":
            text.set_fontweight("bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save(fig, figures / f"end_to_end_{dtype}.pdf")


def plot_template_distribution(rows, figures: Path) -> None:
    if not rows:
        return
    order = {name: index for index, name in enumerate(WORKLOAD_LABELS)}
    rows.sort(key=lambda row: order.get(row["workload"], len(order)))
    widths = (2.15, 1.05, 1.20, 1.20, 1.05, 1.15)
    fig, axis, edges, centers = table_canvas(
        widths,
        len(rows),
        2,
        7.2,
        "3×3×3 template assignment distribution (% of output voxels in K=27 layers).\nAvg Width is the output-row-weighted assigned template width in offsets.",
    )
    top = len(rows) + 2
    table_rule(axis, top, edges[0], edges[-1], 1.2)
    table_rule(axis, len(rows) + 1, edges[1] + 0.06, edges[5] - 0.06, 0.8)
    table_rule(axis, len(rows), edges[0], edges[-1], 0.8)
    table_rule(axis, 0, edges[0], edges[-1], 1.2)
    axis.text(centers[0], len(rows) + 1, "Workload", ha="center", va="center", fontsize=10, fontweight="bold")
    axis.text((edges[1] + edges[5]) / 2, len(rows) + 1.5, "Template Family (%)", ha="center", va="center", fontsize=10, fontweight="bold")
    axis.text(centers[5], len(rows) + 1, "Avg Width\n(offsets)", ha="center", va="center", fontsize=10, fontweight="bold", linespacing=1.15)
    for column, label in enumerate(("center", "skip(2,3)", "skip(1,3)", "full27"), start=1):
        axis.text(centers[column], len(rows) + 0.5, label, ha="center", va="center", fontsize=10, fontweight="bold")
    for row_index, row in enumerate(rows):
        y = len(rows) - row_index - 0.5
        values = (
            WORKLOAD_LABELS.get(row["workload"], row["workload"]),
            f"{float(row['center_percent']):.1f}",
            f"{float(row['skip2_percent']):.1f}",
            f"{float(row['skip1_percent']):.1f}",
            f"{float(row['full27_percent']):.1f}",
            f"{float(row['avg_assigned_width']):.1f}",
        )
        for column, value in enumerate(values):
            axis.text(centers[column], y, value, ha="center", va="center", fontsize=10)
    save(fig, figures / "template_distribution.pdf")


def plot_effective_throughput(rows, figures: Path) -> None:
    if not rows:
        return
    workloads = sorted({row["workload"] for row in rows})
    for workload in workloads:
        selected = [row for row in rows if row["workload"] == workload]
        values = {row["backend"]: row for row in selected}
        if not values:
            continue
        group_x = np.array([0.0, 1.6])
        width = 0.18
        fig, axis = plt.subplots(figsize=(5.5, 2.8))
        for index, backend in enumerate(BACKENDS):
            if backend not in values:
                continue
            offset = (index - 1.5) * (width + 0.04)
            axis.bar(
                group_x + offset,
                [float(values[backend]["raw_tflops"]), float(values[backend]["effective_tflops"])],
                width,
                label=LABELS[backend],
                color=COLORS[backend],
                hatch=HATCHES[backend],
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        axis.set_xticks(group_x, ("Raw", "Effective"))
        axis.set_ylabel("Throughput (TFLOPS)", fontsize=12)
        axis.tick_params(labelsize=11)
        axis.set_xlim(group_x[0] - 0.6, group_x[1] + 0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.25, linewidth=0.3)
        legend = axis.legend(frameon=False, fontsize=11, ncol=2, loc="upper right", handlelength=1.5, columnspacing=1.0)
        for text in legend.get_texts():
            if text.get_text() == "GTSparse":
                text.set_fontweight("bold")
        fig.tight_layout()
        save(fig, figures / f"effective_throughput_{workload}.pdf")


def plot_time_breakdown(rows, figures: Path) -> None:
    if not rows:
        return
    workload_order = {name: index for index, name in enumerate(WORKLOAD_LABELS)}
    backend_order = {name: index for index, name in enumerate(BACKENDS)}
    rows.sort(key=lambda row: (workload_order.get(row["workload"], len(workload_order)), backend_order[row["backend"]]))
    widths = (2.25, 1.45, 1.25, 1.25, 1.15)
    fig, axis, edges, centers = table_canvas(
        widths,
        len(rows),
        1,
        6.4,
        "Builder/kernel latency breakdown (ms). Component shares come from the breakdown runs\nand are scaled so Builder + Kernel equals the corresponding main E2E latency.",
    )
    table_rule(axis, len(rows) + 1, edges[0], edges[-1], 1.2)
    table_rule(axis, len(rows), edges[0], edges[-1], 0.8)
    table_rule(axis, 0, edges[0], edges[-1], 1.2)
    for column, label in enumerate(("Workload", "System", "Builder\n(ms)", "Kernel\n(ms)", "Total\n(ms)")):
        axis.text(centers[column], len(rows) + 0.5, label, ha="center", va="center", fontsize=10, fontweight="bold", linespacing=1.1)
    grouped = []
    for workload in sorted({row["workload"] for row in rows}, key=lambda name: workload_order.get(name, len(workload_order))):
        grouped.append((workload, [row for row in rows if row["workload"] == workload]))
    cursor = 0
    for workload, workload_rows in grouped:
        if cursor:
            table_rule(axis, len(rows) - cursor, edges[0], edges[-1], 0.8)
        group_center = len(rows) - cursor - len(workload_rows) / 2
        axis.text(centers[0], group_center, WORKLOAD_LABELS.get(workload, workload), ha="center", va="center", fontsize=10)
        for row in workload_rows:
            y = len(rows) - cursor - 0.5
            values = (
                LABELS[row["backend"]],
                f"{float(row['builder_median_ms']):.2f}",
                f"{float(row['kernel_median_ms']):.2f}",
                f"{float(row['total_median_ms']):.2f}",
            )
            weight = "bold" if row["backend"] == "gtsparse" else "normal"
            for column, value in enumerate(values, start=1):
                axis.text(centers[column], y, value, ha="center", va="center", fontsize=10, fontweight=weight)
            cursor += 1
    save(fig, figures / "time_breakdown.pdf")


def plot_peak_memory(rows, figures: Path) -> None:
    workloads = sorted({row["workload"] for row in rows})
    if not workloads:
        return
    workload_order = {name: index for index, name in enumerate(WORKLOAD_LABELS)}
    workloads.sort(key=lambda workload: workload_order.get(workload, len(workload_order)))
    widths = (2.15, 1.35, 1.20, 1.05, 1.35)
    fig, axis, edges, centers = table_canvas(
        widths,
        len(workloads),
        1,
        6.8,
        "Peak GPU memory usage (MB), measured with torch.cuda.max_memory_allocated.\nValues include model parameters, inputs, and forward temporary allocations.",
    )
    table_rule(axis, len(workloads) + 1, edges[0], edges[-1], 1.2)
    table_rule(axis, len(workloads), edges[0], edges[-1], 0.8)
    table_rule(axis, 0, edges[0], edges[-1], 1.2)
    for column, label in enumerate(("Workload", "Minkowski", "SpConv", "TS++", "GTSparse")):
        axis.text(centers[column], len(workloads) + 0.5, label, ha="center", va="center", fontsize=10, fontweight="bold")
    for row_index, workload in enumerate(workloads):
        y = len(workloads) - row_index - 0.5
        selected = {row["backend"]: row for row in rows if row["workload"] == workload}
        axis.text(centers[0], y, WORKLOAD_LABELS.get(workload, workload), ha="center", va="center", fontsize=10)
        for column, backend in enumerate(BACKENDS, start=1):
            value = f"{float(selected[backend]['peak_memory_mb']):.1f}" if backend in selected else "--"
            weight = "bold" if backend == "gtsparse" else "normal"
            axis.text(centers[column], y, value, ha="center", va="center", fontsize=10, fontweight=weight)
    save(fig, figures / "peak_memory.pdf")


def plot_ablation(rows, figures: Path) -> None:
    rows = [row for row in rows if row["experiment"] == "ablation" and row["backend"] == "gtsparse" and row["metric"] == "conv_only"]
    if not rows:
        return
    rows.sort(key=lambda row: int(row["min_template"]))
    labels = {
        "0": "All templates",
        "1": "No center",
        "4": "No center\n+ no skip(2,3)",
        "7": "Full27 only",
    }
    colors = ("#fed9a6", "#f9c67a", "#f4a84e", "#d4760a")
    hatches = ("", "//", "\\", "xx")
    values = [float(row["median_ms"]) for row in rows]
    fig, axis = plt.subplots(figsize=(3.2, 3.0))
    for index, value in enumerate(values):
        axis.bar(index, value, 0.6, color=colors[index], hatch=hatches[index], edgecolor="black", linewidth=0.6)
    axis.set_xticks([])
    axis.set_ylabel("Latency (ms)", fontsize=13)
    axis.tick_params(axis="y", labelsize=12)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.25, linewidth=0.3)
    ymin = min(values) * 0.88
    ymax = max(values) * 1.04
    axis.set_ylim(ymin, ymax)
    for index, value in enumerate(values):
        axis.text(index, value + (ymax - ymin) * 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=11)
    handles = [Patch(facecolor=colors[index], hatch=hatches[index], edgecolor="black", linewidth=0.6, label=labels[row["min_template"]]) for index, row in enumerate(rows)]
    axis.legend(handles=handles, fontsize=11, loc="lower left", framealpha=0.9, handlelength=1.0, ncol=2, bbox_to_anchor=(-0.3, 1.02), columnspacing=0.6, handletextpad=0.4)
    fig.tight_layout()
    save(fig, figures / "ablation.pdf")


def plot_sensitivity(rows, figures: Path) -> None:
    rows = [row for row in rows if row["experiment"] == "sensitivity" and row["metric"] == "conv_only"]
    if not rows:
        return
    gpu = sorted({row["gpu"] for row in rows})[0]
    rows = [row for row in rows if row["gpu"] == gpu]
    sweeps = sorted({int(row["sweeps"]) for row in rows})
    fig, axis = plt.subplots(figsize=(2.4, 2.4))
    for backend in BACKENDS:
        selected = sorted((row for row in rows if row["backend"] == backend), key=lambda row: int(row["sweeps"]))
        if selected:
            axis.plot(
                range(len(selected)),
                [float(row["median_ms"]) for row in selected],
                marker=MARKERS[backend],
                color=LINE_COLORS[backend],
                label=LABELS[backend],
                linewidth=1.5,
                markersize=6,
                markeredgewidth=1.2,
                markeredgecolor="black",
                markerfacecolor=LINE_COLORS[backend],
            )
    axis.set_xticks(range(len(sweeps)), [str(value) for value in sweeps], fontsize=11)
    axis.set_xlim(-0.3, len(sweeps) - 0.7)
    axis.set_xlabel("Number of fused sweeps", fontsize=11)
    axis.set_ylabel("Latency (ms)", fontsize=11)
    axis.tick_params(axis="y", labelsize=10)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(alpha=0.25, linewidth=0.3)
    legend = axis.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.4)
    for text in legend.get_texts():
        if text.get_text() == "GTSparse":
            text.set_fontweight("bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    save(fig, figures / "sensitivity.pdf")


def plot_per_frame(rows, figures: Path) -> None:
    target = "voxelnext_nuscenes_sweeps1"
    rows = [row for row in rows if row["workload"] == target and row["metric"] == "conv_only"]
    if not rows:
        return
    gpu = sorted({row["gpu"] for row in rows})[0]
    fig, axis = plt.subplots(figsize=(5.5, 2.2))
    for backend in ("spconv", "torchsparse", "gtsparse"):
        selected = sorted((row for row in rows if row["gpu"] == gpu and row["backend"] == backend), key=lambda row: int(row["frame_index"]))
        if selected:
            latency = np.array([float(row["latency_ms"]) for row in selected])
            window = 20 if len(latency) >= 20 else 1
            smoothed = np.convolve(latency, np.ones(window) / window, mode="valid")
            x = np.arange(window - 1, len(latency))
            marker_every = max(1, len(smoothed) // 5)
            axis.plot(
                x,
                smoothed,
                label=LABELS[backend],
                color=LINE_COLORS[backend],
                linewidth=1.5,
                alpha=0.95,
                marker=MARKERS[backend],
                markevery=marker_every,
                markersize=6,
                markeredgewidth=1.2,
                markeredgecolor="black",
                markerfacecolor=LINE_COLORS[backend],
            )
    axis.set_xlabel("Frame index", fontsize=12)
    axis.set_ylabel("Latency (ms)", fontsize=12)
    axis.tick_params(labelsize=11)
    legend = axis.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.5)
    for text in legend.get_texts():
        if text.get_text() == "GTSparse":
            text.set_fontweight("bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.4)
    axis.set_xlim(0, max(1, max(int(row["frame_index"]) for row in rows)))
    fig.tight_layout()
    save(fig, figures / "per_frame_latency.pdf")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latency = read_csv(args.results / "latency.csv")
    plot_end_to_end(latency, args.figures, "fp16")
    plot_end_to_end(latency, args.figures, "fp32")
    plot_template_distribution(read_csv(args.results / "template_distribution.csv"), args.figures)
    plot_effective_throughput(read_csv(args.results / "effective_throughput.csv"), args.figures)
    plot_time_breakdown(read_csv(args.results / "time_breakdown.csv"), args.figures)
    plot_peak_memory(read_csv(args.results / "peak_memory.csv"), args.figures)
    plot_ablation(latency, args.figures)
    plot_sensitivity(latency, args.figures)
    plot_per_frame(read_csv(args.results / "per_frame_latency.csv"), args.figures)


if __name__ == "__main__":
    main()
