# CVPR 2026 Poster: A Geometric Algebra-Informed 3D Gaussian Splatting Framework for Wireless Scene Representation

This repository contains the code for the CVPR 2026 poster:

**A Geometric Algebra-Informed 3D Gaussian Splatting Framework for Wireless Scene Representation**  
*CVPR 2026 Poster*

---

## Overview

GAI-GS introduces geometric algebra priors into the 3D Gaussian Splatting pipeline for wireless scene modeling and signal representation.
This trimmed project version keeps the RFID reconstruction pipeline and removes the legacy secondary-task path together with generated output and logging artifacts.

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate gaigs
```

Install submodules/extensions:

```bash
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/fused-ssim
```

## Figure Correction

The original CDF-SSIM figure in the paper contains a plotting error.
The corrected figure is available here: [Corrected CDF-SSIM figure](assets/cdf2ssim_corrected.pdf).

## Citation

If you find this repository useful, please cite our CVPR 2026 poster:

```bibtex
To appear
```

## Acknowledgments

This project builds on and benefits from prior work, including:

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- Some code snippets are borrowed from [WRF-GS+](https://github.com/wenchaozheng/WRF-GSplus/tree/main) and [Geometric-Algebra-Transformer](https://github.com/qualcomm-ai-research/geometric-algebra-transformer)
- Project submodules under `./submodules`
