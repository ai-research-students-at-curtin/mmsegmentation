# AI Team's MMSegmentation
This is the Curtin AI Team's fork of MMSegmentation, implementing new models, datasets, and approaches.

## Setup
First, set up your datasets using symlinks (no environment variables are used here!) for example, here's cityscapes:
```sh
ln -s <your_cityscapes_root_path> ./data/cityscapes
```

## Some quick scripts
### SCMNet
Training SCMNet locally
```sh
sh tools/dist_train.sh configs/scmnet/scmnet_cityscapes.py 1
```
Testing SCMNet locally
```sh
sh tools/dist_test.sh configs/scmnet/scmnet_cityscapes.py work_dirs/scmnet_cityscapes/latest.pth 1 --eval mIoU
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
Training SegFormer normally
```sh
# Get the pretrained weights first from https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia, and move to pretrain/mit_b0.pth

sh tools/dist_train.sh configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py 1
```