# GTSparse: A Geometric-Template-Driven Sparse Convolution Runtime on GPUs (Artifact Evaluation)

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
For convenience, we also provide a temporary [OneDrive mirror of `dataset.tar`](https://1drv.ms/u/c/b44f55aa07e438ba/IQBItnLwoagRRrxZpzJcAUD2ATdWtIAN4VN5IeREuDytnn4?e=RkpLuy), an archive containing all three prepared datasets: KITTI, NuScenes, and SemanticKITTI. Download the archive to the repository root and extract it with:

```bash
tar -xf dataset.tar
```

The archive extracts directly to `dataset/`. Users are responsible for complying with the original licenses of KITTI, NuScenes, and SemanticKITTI.

Note: the nuScenes loader resolves the version by probing `v1.0-trainval`, then `v1.0-mini`, then `v1.0-test` under the given root, so different versions (e.g., mini and test) must be placed under separate roots.

## Installation

The artifact uses Python 3.10, PyTorch 2.1.2, and CUDA Toolkit 12.1. Clone the baseline submodules and install all the baselines:

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

`validate_install.py` runs one 3x3x3 SubM convolution layer through GTSparse, SpConv, TorchSparse++, and MinkowskiEngine against a dense PyTorch reference, with TF32 disabled. The comparison is tolerance-based rather than bitwise: engines that accumulate in FP32 (GTSparse, TorchSparse++, MinkowskiEngine) agree at FP32 1e-3 and FP16 2e-2 (atol/rtol); SpConv's FP16 kernels accumulate in FP16 and are checked at atol=1.0. Bitwise equality across engines is not expected, since each engine reduces the kernel offsets in a different order and the production build uses fast-math.

### Container

A thin Dockerfile is provided: the image contains only the CUDA toolkit and OS dependencies, and the environment is built inside a named volume on the first run when the GPU is present during installation, matching the bare-metal path (arch detection, validation). Build the image (fast, no GPU needed):

```bash
bash scripts/build_container.sh
```

The environment is installed on the first run (~40–60 minutes; the host needs `nvidia-container-toolkit` for `--gpus all`). The install already runs `validate_install.py` at the end. `--shm-size=1g` is required: the DataLoader workers share batches through `/dev/shm`, whose container default (64 MB) is too small for multi-sweep point clouds.

```bash
# 1. Install the environment (one time):
docker run --rm --gpus all --shm-size=1g -v gtsparse-workspace:/workspace \
  gtsparse-artifact bash scripts/install.sh
# 2. Run experiments (reuses the installed environment):
docker run --rm --gpus all --shm-size=1g -v gtsparse-workspace:/workspace \
  -v /path/to/dataset:/workspace/GTSparse/dataset \
  gtsparse-artifact bash run_artifact.sh --end-to-end fp16
```

Datasets are intentionally not baked into the image; mount them at runtime. The bare-metal installation above remains the primary installation path.

## Evaluation

### Performance Configuration

Absolute latency can vary slightly across machines even with the same GPU model. Different host platforms have different CPU microarchitectures, memory subsystems, PCIe generations and topologies, motherboard firmware, and cooling capabilities; CPU and GPU power or clock policies introduce further variation. To reduce run-to-run variation, we recommend using a stable performance state and keeping the settings unchanged across all backends. The relative performance trends should nevertheless remain consistent.

On Linux systems that expose CPU frequency and power controls, a performance governor together with a sustainable minimum frequency and package power limit can improve measurement stability. Some `thermald` configurations may restore the default power limit during a long run; when appropriate, it can be stopped temporarily while measurements are collected. The following shows the configuration used by our RTX 3080 test machine, which has an Intel i7-10700:

```bash
sudo systemctl stop thermald
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -d 4.60GHz
echo 125000000 | sudo tee /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw
```

These example values are specific to the test machine. Suitable settings depend on the CPU, firmware, available controls, and cooling system. A short turbo window can otherwise expire during the experiment and raise runtimes that contain host-side GPU launch or synchronization work.

For NVIDIA GPUs, enabling persistence mode and holding a sustainable power limit and clock frequencies constant can further reduce variation. The supported power and clock ranges depend on the GPU and driver. The tested RTX 3080 uses a 340 W limit, a 1800 MHz graphics clock, and a 9501 MHz memory clock:

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 340
sudo nvidia-smi -lgc 1800,1800
sudo nvidia-smi -lmc 9501,9501  # omit if locked memory clocks are unsupported
nvidia-smi --query-gpu=pstate,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv
```

### Experiment Commands

All experiment categories use `run_artifact.sh`:

```bash
bash run_artifact.sh --end-to-end fp16
bash run_artifact.sh --end-to-end fp32
bash run_artifact.sh --end-to-end both
bash run_artifact.sh --microbenchmark
bash run_artifact.sh --ablation
bash run_artifact.sh --sensitivity
bash run_artifact.sh --plots
bash run_artifact.sh --all
```

- `--end-to-end fp16|fp32|both` runs the main workload and backend matrix at the selected precision. FP16 runs GTSparse, SpConv, and TorchSparse++ in FP16 and MinkowskiEngine in FP32; FP32 runs all four systems in FP32. SpConv uses its default sorted bitmask path, and TF32 is disabled in all runs.
- `--microbenchmark` generates the template, throughput, breakdown, stability, and memory measurements.
- `--ablation` runs the template-family ablation.
- `--sensitivity` runs the sweep-sensitivity experiments.
- `--plots` aggregates existing logs and regenerates the result tables and figures without running GPU experiments or requiring a CUDA device.
- `--all` runs every experiment category and then generates all result tables and figures.

Every GPU experiment displays a tqdm progress bar with the total, percentage, and ETA. Completed results are reused by default.

### Resource Requirements

Measured on the reference RTX 3080 with the locked clock settings described above; a faster GPU reduces the GPU time roughly proportionally. Wall time is 10–20% higher than GPU time due to data loading.

| Experiment | Command | GPU time (RTX 3080) | Logs written |
|---|---|---|---|
| End-to-end FP16 | `--end-to-end fp16` | ~1.3 h | ~13 MB |
| End-to-end FP32 | `--end-to-end fp32` | ~0.9 h | ~4 MB |
| Microbenchmark | `--microbenchmark` | ~5 min | ~12 MB |
| Ablation | `--ablation` | ~20 min | ~5 MB |
| Sensitivity | `--sensitivity` | ~2.4 h | ~18 MB |
| All categories | `--all` | ~5 h (~6 h wall) | ~52 MB |

Installation takes about 40 minutes, plus several minutes for SpConv's first-import JIT compilation. The extracted datasets occupy about 200 GB (KITTI ~39 GB, nuScenes ~72 GB, SemanticKITTI ~90 GB); downloading and extracting `dataset.tar` requires roughly twice the extracted size (~400 GB of free space).

### Example Runs

Each experiment category has an example run that completes in a few minutes on any GPU. The examples use the first N frames of the downloaded datasets and write everything under `examples/` (via `--output-root`), leaving `logs/` untouched. The reference outputs of exactly these commands, captured on the RTX 3080, are stored in `examples/logs/` for comparison; see `examples/README.md` for the commands, the expected files, and what to compare (structure, orderings, and magnitudes).

```bash
bash run_artifact.sh --end-to-end fp16 --frames 5 --output-root examples
bash run_artifact.sh --end-to-end fp32 --frames 5 --output-root examples
bash run_artifact.sh --microbenchmark --micro-frames 10 --memory-frames 5 --frames 50 --output-root examples
bash run_artifact.sh --ablation --frames 50 --output-root examples
bash run_artifact.sh --sensitivity --frames 50 --output-root examples
```

### Measurement Methodology

The microbenchmark experiment produces template-family distributions, effective-throughput data, builder/kernel breakdowns, and peak allocated GPU memory. Template-family percentages and average width aggregate output rows across all `3x3x3` (`K=27`) layers on the complete split; layers with other kernel volumes are excluded from this table. Useful work and issued work are reconstructed from a separate 100-frame per-layer profile; SpConv's issued work uses the M-tile width returned by its autotuned kernel for each layer. Breakdown uses the same fixed random sample of 100 frames for every backend. GTSparse uses native builder/kernel CUDA events, while baseline breakdown uses one cold-to-warm pair per independent frame. Aggregation takes the median per-frame component shares and scales them by the overall end-to-end latency, so the displayed builder and kernel values sum to the complete model latency (Table 3 in the paper). Peak memory reports `torch.cuda.max_memory_allocated` from model construction through the measured forwards, with each backend run in a separate process. Ablation evaluates `min_template=0,1,4,7` on VoxelNeXt with 10 sweeps. Sensitivity evaluates all systems with 1, 5, 10, and 20 sweeps.

### Outputs and Paper Reproduction

The complete reference logs collected on our NVIDIA GeForce RTX 3080 are available in the [`artifact-logs-rtx3080-v1` release](https://github.com/eurosys-27-1183/GTSparse/releases/tag/artifact-logs-rtx3080-v1). From the repository root, download and extract the logs, then regenerate all result tables and figures with:

```bash
curl -fL https://github.com/eurosys-27-1183/GTSparse/releases/download/artifact-logs-rtx3080-v1/gtsparse-artifact-logs-rtx3080.tar.gz -o gtsparse-artifact-logs-rtx3080.tar.gz && tar -xzf gtsparse-artifact-logs-rtx3080.tar.gz
bash run_artifact.sh --plots
```

The complete artifact evaluation, including all measurements, result tables, and paper figures, can be reproduced with one command:

```bash
bash run_artifact.sh --all
```

The same work can be run by category when separate commands are more convenient:

```bash
bash run_artifact.sh --end-to-end fp16
bash run_artifact.sh --microbenchmark
bash run_artifact.sh --ablation
bash run_artifact.sh --sensitivity
```

Raw measurements are written under `logs/`; microbenchmark cases are separated by GPU, dtype, and workload so measurements from multiple platforms can coexist. Aggregated paper data from all available platforms is written as CSV files under `results/`. The cross-platform end-to-end figure is written directly under `figures/`, while all platform-specific tables and figures are written under `figures/<gpu>/`. After the measurements are available, `bash run_artifact.sh --plots` regenerates both `results/` and `figures/` directly from the raw logs.

## Citation

If you use this artifact in research, please cite the paper (machine-readable metadata is also provided in `CITATION.cff`):

```bibtex
@inproceedings{gtsparse2027,
  author    = {TBD},
  title     = {GTSparse: A Geometric-Template-Driven Sparse Convolution Runtime for GPUs},
  booktitle = {Proceedings of the 22nd European Conference on Computer Systems (EuroSys '27)},
  pages     = {TBD},
  publisher = {ACM},
  address   = {New York, NY, USA},
  year      = {2027},
  doi       = {TBD},
}
```

The author list, page numbers, and DOI are filled in upon publication.

## License

GTSparse is licensed under the BSD 3-Clause License (see `LICENSE`), which permits use, comparison, and extension with attribution. The baseline engines under `third_parties/` retain their own licenses: cumm and SpConv are Apache-2.0, TorchSparse++ is MIT, sparsehash is BSD-3-Clause, and MinkowskiEngine (the CUDA-13-compatible fork) is MIT. The datasets are distributed under their respective original licenses; users are responsible for complying with them.
