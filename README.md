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
- `--plots` aggregates existing logs and regenerates the result tables and figures without running GPU experiments or requiring a CUDA device.
- `--all` runs every experiment category and then generates all result tables and figures.

Every GPU experiment displays a tqdm progress bar with the total, percentage, and ETA. Completed results are reused by default.

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
