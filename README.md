# PAGAE: Improving Graph Autoencoder by Dual Enhanced Adversary (CogSci 2023)

## PAGAE:

PAGAE is the code for our paper "PAGAE: Improving Graph Autoencoder by Dual Enhanced Adversary", which is published in CogSci 2023. 

<p align="center">
  <img src="./img/model.jpg" width="50%" />
</p>


## Citation
```
@inproceedings{wang2023pagae,
  title={PAGAE: Improving Graph Autoencoder by Dual Enhanced Adversary.},
  author={Wang, Gongju and Li, Mengyao and Feng, Hanbin and Yan, Long and Song, Yulun and Li, Yang and Song, Yinghao},
  booktitle={CogSci},
  year={2023}
}
```

## Data

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/ and here (in a different format): https://github.com/kimiyoung/planetoid

## Models

## Overview
Here we provide an implementation of PAGAE/PAGAEpo in PyTorch, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `data/` contains the necessary dataset files;
- `results/` contains the embedding results;
- `layers.py` contains the implementation of a GCN layer;
- `utils.py` contains the necessary processing function.
- `model.py` contains the implementation of a GAE model, discriminator model and mutual information estimator model.
- `optimizer.py` contains the implementation of the reconstruction loss.

### Usage
For node clustering task:
```
$ python train.py --dataset_str cora --hidden1 128 --hidden2 64 --M 2 --epochs 100
$ python train.py --dataset_str citeseer --hidden1 128 --hidden2 64 --M 2 --epochs 200
$ python train.py --dataset_str pubmed --hidden1 128 --hidden2 64 --M 2 --epochs 200
```

For link prediction task:
```
$ python train_linkpred.py --dataset_str cora --hidden1 128 --hidden2 64 --M 0 --epochs 100
$ python train_linkpred.py --dataset_str citeseer --hidden1 128 --hidden2 64 --M 0 --epochs 200
$ python train_linkpred.py --dataset_str pubmed --hidden1 128 --hidden2 64 --M 0 --epochs 200
```


