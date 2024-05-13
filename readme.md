# Intrinsic Subgraph Generation for Interpretable Graph based Visual Question Answering

Implementation will be available soon.

## Paper Links
[![arXiv](https://img.shields.io/badge/arXiv-2403.17647-b31b1b.svg?style=flat)](https://arxiv.org/abs/2403.17647)
[![Generic badge](https://img.shields.io/badge/LREC_COLING-COMING_SOON-GREEN.svg)](https://shields.io/)

## Approach
![Architecture](./Architecture.jpg)

## Installation
### Python Environment
Create a virtual python environment with e.g. conda:
```bash
conda create --name isubgvqa python=3.11
```
Activate the environment
```bash
conda activate isubgvqa
```
### PyTorch
Please [install PyTorch](https://pytorch.org/get-started/locally/)
```bash
pip install torch torchvision torchaudio
```

### PyG (PyTorch-Geometric)
Please [install PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) 
```bash
pip install torch_geometric
```
Install optional packages:
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### Spacy
Install spacy and download en_core_web_sm
```bash
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```

## Citation
```bibtex
@article{tilli2024intrinsic,
  title={Intrinsic Subgraph Generation for Interpretable Graph based Visual Question Answering},
  author={Tilli, Pascal and Vu, Ngoc Thang},
  journal={arXiv preprint arXiv:2403.17647},
  year={2024}
}
```
