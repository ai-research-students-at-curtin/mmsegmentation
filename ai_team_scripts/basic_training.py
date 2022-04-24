## NOTES
"""
Before you start, run the following:
```sh
mkdir checkpoints
wget https://download.openmmlab.com/mmsegmentation/v0.5/fast_scnn/fast_scnn_lr0.12_8x4_160k_cityscapes/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth -P checkpoints
```
"""

# Load in our config file. This is where we define *everything* about how we want to run training
import os.path as osp
import json # For debugging

import mmcv
from mmcv import Config
from mmseg.datasets.builder import DATASETS
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets.custom import CustomDataset

from model_training import dataset_getters


CAMVID_ROOT = '/home/moritz/Documents/datasets/camvid'
FASTSCNN_CFG_PATH = '../configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py'
FASTSCNN_CHECKPOINT_PATH = 'checkpoints/fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth'

def define_dataset(classes, palette):
    @DATASETS.register_module()
    class CamvidDataset(CustomDataset):
        CLASSES = classes
        PALETTE = palette

        def __init__(self, **kwargs):
            super().__init__(img_suffix='.png', seg_map_suffix='_L_labelIds_11.png', **kwargs)
            assert osp.exists(self.img_dir)


def modify_config(cfg:Config, dataset_getter:dataset_getters.DatasetGetter, dataset_root:str):
    # Since we use only one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg

    for head in cfg.model.auxiliary_head:
        head.norm_cfg = cfg.norm_cfg
        head.num_classes = dataset_getter.get_num_classes()

    cfg.model.decode_head.num_classes = dataset_getter.get_num_classes()

    cfg.dataset_type = 'CamvidDataset'
    cfg.data_root = dataset_root

    cfg.data.samples_per_gpu = 4
    cfg.data.workers_per_gpu = 1

    # NOTE need to look into modifying "img_norm_cfg" for non-pretrained datasets, don't really know what it is/does
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(640, 896), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 896),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = 'train'
    cfg.data.train.ann_dir = 'train_labels'
    cfg.data.train.pipeline = cfg.train_pipeline

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = 'val'
    cfg.data.val.ann_dir = 'val_labels'
    cfg.data.val.pipeline = cfg.test_pipeline

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = 'test'
    cfg.data.test.ann_dir = 'test_labels'
    cfg.data.test.pipeline = cfg.test_pipeline

    # Use pre-trained weights as starting point
    cfg.load_from = FASTSCNN_CHECKPOINT_PATH

    cfg.work_dir = './work_dirs/tutorial'

    cfg.runner.max_iters = 200
    cfg.log_config.interval = 10
    cfg.evaluation.interval = 200
    cfg.checkpoint_config.interval = 200

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    return cfg


def main():
    cfg = Config.fromfile(FASTSCNN_CFG_PATH)
    camvid_getter = dataset_getters.Camvid11(CAMVID_ROOT)

    # Define custom CamVid dataset
    class_names = [c.name for c in camvid_getter.get_classes()]
    palette = camvid_getter.get_colors()
    define_dataset(class_names, palette)

    cfg = modify_config(cfg, camvid_getter, CAMVID_ROOT)

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # For visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
                    meta=dict())


if __name__ == '__main__':
    main()