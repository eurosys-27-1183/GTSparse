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

- `--end-to-end fp16|fp32|both` runs the main workload and backend matrix at the selected precision.
- `--microbenchmark` generates the template, throughput, breakdown, stability, and memory measurements.
- `--ablation` runs the template-family ablation.
- `--sensitivity` runs the sweep-sensitivity experiments.
- `--plots` aggregates existing logs and regenerates the result tables and figures without running GPU experiments.
- `--all` runs every experiment category and then generates all result tables and figures.

The common optional parameters are:

- `--frames N` controls the number of frames for end-to-end, template-profile, ablation, and sensitivity experiments. The default `0` uses the complete dataset split.
- `--micro-frames N` controls sampled timing, profiling, and breakdown measurements; the default is `100`.
- `--memory-frames N` controls peak-memory measurements; the default is `20`.
- `--warmup N` sets the number of global warmup frames; the default is `20`.
- `--timing-repeats N` sets the measured repetitions per frame; the median is recorded and the default is `3`.
- `--device DEVICE` selects the CUDA device; the default is `cuda:0`.
- `--overwrite` replaces completed outputs instead of reusing them.

Every GPU experiment displays a tqdm progress bar with the total, percentage, and ETA. Completed results are reused by default.

### Outputs and Paper Reproduction

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

Raw measurements are written under `logs/`. Aggregated paper data is written as CSV files under `results/`, and the corresponding PDF tables and figures are written under `figures/`. After the measurements are available, `bash run_artifact.sh --plots` regenerates both `results/` and `figures/` directly from the raw logs.
