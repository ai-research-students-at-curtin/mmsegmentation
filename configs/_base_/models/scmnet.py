# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, momentum=0.01)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SCMNet',
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='SCMNetHead',
        num_classes=19,
        in_channels=64,
        channels=48,
        in_index=-1,
        kernel_size=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))