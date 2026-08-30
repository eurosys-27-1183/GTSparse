# GTSparse: A Geomtric-Template-Driven Sparse Convolution Runtime on GPUs

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
python scripts/smoke_test.py
```

## Evaluation

Run evaluation:
```bash
bash run_e2e_v2.sh [backend] [dtype]
```
 + Choices of [backend]: gtsparse, spconv, torchsparse, minkowski.
 + Choices of [dtype]: fp32, fp16. 

Results are saved in `logs/` directory. Note that minkowski backend does not support fp16.
