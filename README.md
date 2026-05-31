# GTSparse: A Geomtric-Template-Driven Sparse Convolution Runtime on GPUs

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

Requires CUDA Toolkit 12.1 with PyTorch 2.1.0 or above.

### GTSparse
```bash
BUILD_MODE=production python3 setup.py develop --user
```

### Other Backends

You would need to install the baseline backends manually. Links:
 - [SpConv v2](https://github.com/traveller59/spconv)
 - [TorchSparse++](https://github.com/mit-han-lab/torchsparse)
 - [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine)

Note that the official Minkowski Engine does not support CUDA 12 and above. We used a third-party updated version [MinkowskiEngineCuda13](https://github.com/AzharSindhi/MinkowskiEngineCuda13)

## Evaluation

Install dependencies:
```bash
pip install -r requirements.txt
```

Run evaluation:
```bash
bash run_e2e_v2.sh [backend] [dtype]
```
 + Choices of [backend]: gtsparse, spconv, torchsparse, minkowski.
 + Choices of [dtype]: fp32, fp16. 

Results are saved in `logs/` directory. Note that minkowski backend does not support fp16.