# A Geometric Algebra-Informed 3D Gaussian Splatting Framework for Wireless Scene Representation Scene Representation

This repository contains the official code for:

**A Geometric Algebra-Informed 3D Gaussian Splatting Framework for Wireless Scene Representation**  
*(to appear in CVPR 2026)*

---

## Overview

GAI-GS introduces geometric algebra priors into the 3D Gaussian Splatting pipeline for wireless scene modeling and signal representation.  
The framework targets wireless field reconstruction and prediction tasks (including BLE/RFID settings) with improved geometric consistency and practical training efficiency.

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate wrfgsplus
```

Install submodules/extensions:

```bash
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/fused-ssim
```

## Data

The code currently supports:

- `rfid` dataset mode (default data root in code: `./data_test200`)
- `ble` dataset mode (default data root in code: `data/Second_room/5GHz/BLE`)

For BLE mode, expected files under the dataset directory include:

- `tx_pos.csv`
- `gateway_rssi.csv`
- `gateway_position.yml`
- `train_index.txt` / `test_index.txt` (auto-generated if needed)

## Training

### Single-GPU

```bash
python train.py --dataset_type rfid --gpu 0
```

```bash
python train.py --dataset_type ble --gpu 0
```

### DDP (multi-GPU)

```bash
bash run_ddp.sh 2
```

You can pass extra training arguments after GPU count, for example:

```bash
bash run_ddp.sh 2 --dataset_type ble
```

## Outputs

- Model checkpoints and point clouds are written to `./output/<timestamp>/`
- Logs are written to `./logs/<timestamp>/`

## Citation

If you find this repository useful, please cite our CVPR 2026 paper:

```bibtex
To appear
```

## Acknowledgments

This project builds on and benefits from prior work, including:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- Some code snippets are borrowed from [WRF-GS+](https://github.com/wenchaozheng/WRF-GSplus/tree/main) and [Geometric-Algebra-Transformer](https://github.com/qualcomm-ai-research/geometric-algebra-transformer)
- Project submodules under `./submodules`
