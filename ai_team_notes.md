# AI Team's MMSegmentation
This is the Curtin AI Team's fork of MMSegmentation, implementing new models, datasets, and approaches.

## Setup
First, (obviously,) clone this repository.

### Datasets setup
Set up your datasets using symlinks (no environment variables are used here!). For example, here's cityscapes:
```sh
ln -s <your_cityscapes_root_path> ./data/cityscapes
```

### Conda environment setup
```sh
conda create -n open-mmlab python=3.10 -y
conda activate open-mmlab

conda install pytorch=1.11.0 torchvision cudatoolkit=10.2 -c pytorch
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11/index.html

pip install -e .
```

## Simple Example - Inference on Fast-SCNN
First, get the weights [here](https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth), and save them to `./pretrained/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth`.

```sh
python3 tools/test.py configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py ./pretrained/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth --eval mIoU
```

## Some quick scripts
### SCMNet
Training SCMNet locally
```sh
sh tools/dist_train.sh configs/scmnet/scmnet_cityscapes.py 1
```
Testing SCMNet locally
```sh
python3 tools/test.py configs/scmnet/scmnet_cityscapes.py work_dirs/scmnet_cityscapes/latest.pth --eval mIoU \
--cfg-options norm_cfg.type='BN' model.backbone.norm_cfg.type='BN' model.decode_head.norm_cfg.type='BN' \
--show # Optionally (shows each prediction)
```
(Same as above but show visualisations)
```sh
sh tools/dist_test.sh configs/scmnet/scmnet_cityscapes.py work_dirs/scmnet_cityscapes/latest.pth 1 --show
```

Training SCMNet on Pawsey
```sh
GPUS_PER_NODE=1 CPUS_PER_TASK=4 GPUS=1 sh tools/slurm_train.sh gpuq-dev mms-scmnet-test configs/scmnet/scmnet_cityscapes.py --work-dir work_dirs/scmnet_cityscapes
```

### SegFormer
#### **Training SegFormer normally**
(first get the pretrained weights first from https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia, and move to `pretrain/mit_b0.pth`)
```sh
sh tools/dist_train.sh configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py 1
```