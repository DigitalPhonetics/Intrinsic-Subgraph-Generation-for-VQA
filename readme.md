# Intrinsic Subgraph Generation for Interpretable Graph based Visual Question Answering

## Paper Links

**Discrete Subgraph Sampling** \
[![arXiv](https://img.shields.io/badge/arXiv-2412.08263-b31b1b.svg?style=flat)](https://arxiv.org/abs/2412.08263)
[![Generic badge](https://img.shields.io/badge/COLING-Link-GREEN.svg)]()

**Intrinsic Subgraph Generation** \
[![arXiv](https://img.shields.io/badge/arXiv-2403.17647-b31b1b.svg?style=flat)](https://arxiv.org/abs/2403.17647)
[![Generic badge](https://img.shields.io/badge/LREC_COLING-Link-GREEN.svg)](https://aclanthology.org/2024.lrec-main.806)

## Approach
![Architecture](./Architecture.jpg)

## Documentation
[Documentation](https://intrinsic-subgraph-generation-for-interpretable-graph-based-vis.readthedocs.io/en/latest/index.html)

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

## Citations
```bibtex
@article{tilli2024discretesubgraphsamplinginterpretable,
      title={Discrete Subgraph Sampling for Interpretable Graph based Visual Question Answering}, 
      author={Pascal Tilli and Ngoc Thang Vu},
      year={2024},
      eprint={2412.08263},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.08263}, 
}

@inproceedings{tilli-vu-2024-intrinsic-subgraph,
    title = "Intrinsic Subgraph Generation for Interpretable Graph Based Visual Question Answering",
    author = "Tilli, Pascal  and Vu, Ngoc Thang",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italy",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.806",
    pages = "9204--9223",
}
```

## GitHub References
**IMLE** \
https://github.com/uclnlp/torch-imle

**AIMLE** \
https://github.com/EdinburghNLP/torch-adaptive-imle

**Gumbel / Simple** \
https://github.com/chendiqian/PR-MPNN/
