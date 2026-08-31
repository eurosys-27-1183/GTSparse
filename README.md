# GTSparse: A Geometric-Template-Driven Sparse Convolution Runtime on GPUs

## Artifact Evaluation

## Dataset Preparation

You would need to download KITTI, NuScenes, and SemanticKITTI datasets. Links:
 - [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
 - [NuScenes](https://www.nuscenes.org/nuscenes)
 - [SemanticKITTI](http://semantic-kitti.org/)

These datasets should be organized as follows (create a `dataset` directory and place the datasets in it):

```
dataset
├── kitti
│   ├── testing
│   └── training
├── nuscenes
│   ├── LICENSE
│   ├── maps
│   ├── samples
│   ├── sweeps
│   └── v1.0-test
└── semantickitti
    ├── README
    └── dataset
        └── sequences
```
## Installation

The artifact uses Python 3.10, PyTorch 2.1.2, and CUDA Toolkit 12.1. Clone the baseline submodules before installation:

```bash
git submodule update --init --recursive
bash scripts/install.sh
source scripts/activate.sh
```

The installer creates `.venv`, installs the Ubuntu build dependencies, installs PyTorch cu121, and builds cumm, SpConv, TorchSparse++, MinkowskiEngine, and GTSparse from source. The first SpConv import compiles its generated CUDA sources and can take several minutes. The installer uses `/usr/local/cuda-12.1` when available. If CUDA Toolkit 12.1 is not installed, it installs the toolkit without a driver under `.cuda/cuda-12.1`.

Run the activation script before building or evaluating in a new shell:

```bash
source scripts/activate.sh
python scripts/validate_install.py
```

## Evaluation

The artifact exposes one experiment entry point:

```bash
bash run_artifact.sh --end-to-end fp16
bash run_artifact.sh --end-to-end fp32
bash run_artifact.sh --microbenchmark
bash run_artifact.sh --ablation
bash run_artifact.sh --sensitivity
```

The FP16 end-to-end experiment runs GTSparse, SpConv, and TorchSparse++ in FP16 and MinkowskiEngine in FP32. The FP32 experiment runs all four systems in FP32. SpConv uses its default sorted bitmask path, TorchSparse++ and GTSparse run without sorting, and TF32 is disabled. End-to-end evaluation covers SECOND/KITTI, VoxelNeXt/nuScenes with 1 and 10 sweeps, and MinkUNet42/SemanticKITTI. Detection workloads report sparse-convolution latency and MinkUNet42 reports end-to-end latency.

The microbenchmark experiment produces template-family distributions, effective-throughput data, builder/kernel breakdowns, and peak allocated GPU memory. Template-family percentages and average width aggregate output rows across all `3x3x3` (`K=27`) layers on the complete split; layers with other kernel volumes are excluded from this table. Useful work and issued work are reconstructed from a separate 100-frame per-layer profile; SpConv's issued work uses the M-tile width returned by its autotuned kernel for each layer. GTSparse breakdown uses native builder/kernel CUDA events, while baseline breakdown uses one cold-to-warm pair per independent frame. Aggregation takes the median per-frame component shares and scales them by the overall end-to-end latency, so the displayed builder and kernel values sum to the complete model latency. Peak memory reports `torch.cuda.max_memory_allocated` from model construction through the measured forwards, with each backend run in a separate process. Ablation evaluates `min_template=0,1,4,7` on VoxelNeXt with 10 sweeps. Sensitivity evaluates all systems with 1, 5, 10, and 20 sweeps.

Run every experiment and generate all tables and figures with:

```bash
bash run_artifact.sh --all
```

The end-to-end, template-distribution, ablation, and sensitivity experiments use the complete dataset split by default. `--frames N` selects the first `N` measured frames without changing the experiment path. Throughput and time-breakdown measurements use 100 frames by default and accept `--micro-frames N`; peak-memory measurement uses 20 frames and accepts `--memory-frames N`. Timing defaults to 20 global warmup frames, two local warmup repetitions, and the median of three measured repetitions.

Raw measurements are written under `logs/`. `results/` contains CSV tables aggregated from those logs, and `figures/` contains plots generated only from the CSV tables. Reprocess existing logs without running GPU experiments with:

```bash
bash run_artifact.sh --plots
```

Every GPU measurement displays a per-frame tqdm progress bar with the known total, percentage, and ETA. Completed results are reused by default: fixed-frame outputs must contain the requested number of records, while a full-split run is reusable only after its summary has been written at the end of the split. The runner prints each reused summary or JSONL path with a `[reuse]` prefix. Force every selected experiment to overwrite existing logs with:

```bash
bash run_artifact.sh --all --overwrite
```
