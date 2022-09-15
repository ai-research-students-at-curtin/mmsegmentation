This document tracks everything that might be valuable to remember later.

## General Setup
Install miniconda (on my Linux, I installed using the source sh script.)

Create a new conda environment (python 3.8)

```sh
conda create --name udaseg python=3.8
```

Install pytorch
```sh
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

Install mmcv and mmsegmentation locally
```sh
# mmcv
pip install -U openmim
mim install mmcv-full

# mmsegmentation (local) - run while inside the mmsegmentation directory
pip install -v -e .
```

# Experiments
## Experiment 1 - Zero-Shot Inference
As a baseline experiment, we investigate the transferrability of SegFormer using zero-shot predictions - only training on $DATASET_1, predictions on $DATASET_2.

### Converting ImageNet SegFormer weights
```sh
export PRETRAIN_PATH='/home/moritz/Documents/github/mmsegmentation/ai_team_scripts/weights/mit_b5.pth'
export STORE_PATH='/home/moritz/Documents/github/mmsegmentation/ai_team_scripts/weights/mit_b5_pretrain.pth'
python tools/model_converters/mit2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

## Experiment 2 - Class-Balanced Self-Training
We follow the state-of-the-art work of *[Zou et al. (2016)](https://arxiv.org/abs/1810.07911)* to investigate the quality of pseudolabels produced by SegFormer.

## Experiment 3 - Siamese network with MMD loss 

## Experiment 4 - Bidirectional Cross-Stream SegFormer
We adapt the work of *[Wang et al. (2022)](https://arxiv.org/abs/2201.05887)* to the semantic segmentation context, applying Source-Target and Target-Source models which use cross-attention to produce confusion between domains.


