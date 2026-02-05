# GAI-GS: Geometric Algebra Integrated Gaussian Splatting for Wireless Signal Reconstruction

This repository implements a novel approach for RFID spectrum reconstruction using 3D Gaussian Splatting combined with Geometric Algebra Transformers (GATr).

## Features

- **3D Gaussian Splatting** for wireless signal field representation
- **GATr Encoder** with directional cross-attention for geometric-aware signal processing
- **RFID Spectrum Reconstruction** from sparse measurements
- **DDP Support** for distributed training across multiple GPUs

## Installation

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate gai-gs
```

### 2. Install Submodules

```bash
# Install diff-gaussian-rasterization
cd submodules/diff-gaussian-rasterization
pip install .
cd ../..

# Install simple-knn
cd submodules/simple-knn
pip install .
cd ../..

# Install fused-ssim
cd submodules/fused-ssim
pip install .
cd ../..
```

## Usage

### Single GPU Training

```bash
python train.py -s <path_to_data> --data_type rfid
```

### Multi-GPU DDP Training

```bash
# Using --use_ddp flag with specified GPUs
python train.py -s <path_to_data> --data_type rfid --use_ddp --gpus "0,1,2,3"

# Or using torchrun
torchrun --nproc_per_node=4 train.py -s <path_to_data> --data_type rfid --use_ddp
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-s, --source_path` | Path to dataset | Required |
| `--data_type` | Data type (rfid) | rfid |
| `--use_ddp` | Enable distributed training | False |
| `--gpus` | GPU list (e.g., "0,1,2,3") | None |
| `--world_size` | Number of GPUs for DDP | 1 |
| `--iterations` | Number of training iterations | 100000 |
| `--model_path` | Path to save/load model | auto-generated |

## Project Structure

```
GAI-GS/
├── train.py                 # Main training script
├── arguments/               # Command-line argument parsing
├── gaussian_renderer/       # Gaussian rasterization renderer
├── scene/
│   ├── __init__.py         # Scene setup and data loading
│   ├── gaussian_model.py   # Gaussian model with MappingNetwork
│   ├── mapping_network.py  # Signal mapping network
│   ├── dataloader.py       # Dataset classes
│   └── cameras.py          # Camera utilities
├── gatr/                   # Geometric Algebra Transformer
│   ├── layers/             # GATr layers (attention, MLP, etc.)
│   ├── nets/               # GATr network architectures
│   ├── primitives/         # GA primitives (products, norms)
│   ├── interface/          # GA object embeddings
│   └── utils/              # Utility functions
├── utils/                  # General utilities
└── submodules/             # External dependencies
    ├── diff-gaussian-rasterization/
    ├── simple-knn/
    └── fused-ssim/
```

## Citation

If you find this work useful, please cite:

```will update
```

## Acknowledgments

This project builds upon:
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [GATr: Geometric Algebra Transformer](https://github.com/Qualcomm-AI-research/geometric-algebra-transformer)
- [WRF-GS](https://github.com/wenchaozheng/WRF-GS)
