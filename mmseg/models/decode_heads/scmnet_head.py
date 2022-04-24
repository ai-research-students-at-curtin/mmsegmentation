# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import DepthwiseSeparableConvModule

from ..builder import HEADS
from .fcn_head import FCNHead


@HEADS.register_module()
class SCMNetHead(FCNHead):
    """Depthwise-Separable Fully Convolutional Network for Semantic
    Segmentation.

    This head is implemented according to `SCMNet: Shared Context Mining 
    Network for Real-time Semantic Segmentation 
    <https://doi.org/10.1109/DICTA52665.2021.9647401>`_.

    Args:
        in_channels(int): Number of output channels of DS-FFM.
        channels(int): Number of middle-stage channels in the decode head.
        concat_input(bool): Whether to concatenate original decode input into
            the result of several consecutive convolution layers.
            Default: True.
        num_classes(int): Used to determine the dimension of
            final prediction tensor.
        in_index(int): Correspond with 'out_indices' in FastSCNN backbone.
        norm_cfg (dict | None): Config of norm layers.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_decode(dict): Config of loss type and some
            relevant additional options.
        dw_act_cfg (dict):Activation config of depthwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: None.
    """

    def __init__(self, dw_act_cfg=None, **kwargs):
        super(SCMNetHead, self).__init__(**kwargs)
        self.convs[0] = DepthwiseSeparableConvModule(
            self.in_channels,
            64,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            norm_cfg=self.norm_cfg,
            dw_act_cfg=dw_act_cfg)

        self.convs[1] = DepthwiseSeparableConvModule(
            64,
            48,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            norm_cfg=self.norm_cfg,
            dw_act_cfg=dw_act_cfg)
