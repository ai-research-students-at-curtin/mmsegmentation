import torch
import torch.nn as nn
from torch.nn import functional
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, build_upsample_layer, build_activation_layer
from mmcv.runner import BaseModule
from mmcv.cnn.bricks import Swish

from mmseg.models.decode_heads.psp_head import PPM
from ..builder import BACKBONES
from ..utils import InvertedResidual

class ContextMiningModule(nn.Module):
    """Context mining module (CMM)."""
    def __init__(self,
                 channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(ContextMiningModule, self).__init__()
        assert channels % 4 == 0

        mid_channels = channels // 4

        self.pointwise = ConvModule(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=(1,1),
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        
        self.dilated_depthwise = DepthwiseSeparableConvModule(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=(3,3),
            stride=1,
            dilation=6,
            padding='same',
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.feature_pool = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(2,2),
                stride=2),
            ConvModule(
                in_channels=channels,
                out_channels=mid_channels,
                kernel_size=(1,1),
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            build_upsample_layer(
                cfg=dict(type='bilinear'),
                scale_factor=(2,2))
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(
                output_size=(1,1)
            ),
            ConvModule(
                in_channels=channels,
                out_channels=mid_channels,
                kernel_size=(1,1),
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            # NOTE: Upsampling needs to be done here but we don't 
            #   have a layer for it - so we do it in `forward()`
        )


    def forward(self, x):
        upsample_shape = (x.size()[-2], x.size()[-1])
        x_pointwise = self.pointwise(x)
        x_dilated_depthwise = self.dilated_depthwise(x)
        x_feature_pool = self.feature_pool(x)
        x_global_pool = self.global_pool(x)
        x_global_pool = functional.upsample_bilinear(x_global_pool, upsample_shape)
        
        x_concat = torch.cat((
                x_pointwise,
                x_dilated_depthwise,
                x_feature_pool,
                x_global_pool), 
            dim=1)
        
        return x_concat

class DeepShallowFeatureFusionModule(nn.Module):
    """Deep-shallow feature fusion module"""
    def __init__(self,
            channel_sizes = [24, 32, 48, 96, 160, 160],
            intermediate_channels=64,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish')
            ):
        super(DeepShallowFeatureFusionModule, self).__init__()

        # Input Stage
        self.inputs = []
        for ii in range(6):
             self.inputs.append(
                ConvModule(
                    in_channels=channel_sizes[ii],
                    out_channels=intermediate_channels,
                    kernel_size=(1,1),
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None)
                )

        self.f6_intermed = nn.Sequential(
            build_activation_layer(cfg=act_cfg),
            DepthwiseSeparableConvModule(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                kernel_size=3,
                stride=1,
                padding=1)
            )
        self.f2_intermed_1 = nn.Sequential(
            build_activation_layer(cfg=act_cfg),
            DepthwiseSeparableConvModule(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                kernel_size=3,
                stride=1,
                padding=1)
            )  
        self.f2_intermed_2 = nn.Sequential(
            build_activation_layer(cfg=act_cfg),
            DepthwiseSeparableConvModule(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                kernel_size=3,
                stride=1,
                padding=1)
            )  


        # Add sets of 3 identical intermediary layers for F3, F4, & F5
        self.f5_intermeds = []
        self.f4_intermeds = []
        self.f3_intermeds = []
        extended_intermeds = [
            self.f5_intermeds,
            self.f4_intermeds,
            self.f3_intermeds,]
        
        for intermed in extended_intermeds:
            for ii in range(3):
                intermed.append(
                    nn.Sequential(
                        build_activation_layer(cfg=act_cfg),
                        DepthwiseSeparableConvModule(
                            in_channels=intermediate_channels,
                            out_channels=intermediate_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1)
                        ))
        
        self.out = nn.Sequential(
            build_activation_layer(cfg=act_cfg),
            DepthwiseSeparableConvModule(
                in_channels=intermediate_channels,
                out_channels=intermediate_channels,
                kernel_size=3,
                stride=1,
                padding=1)
            )

        # Force modules in lists into cuda 
        all_list_modules = self.inputs + self.f5_intermeds + self.f4_intermeds + self.f3_intermeds
        for m in all_list_modules:
            m.cuda()


    def forward(self, f1, f2, f3, f4, f5, f6):
        features = [f1, f2, f3, f4, f5, f6]
        result_features = []

        # Refer to the paper's Figure 3 for help
        for ii in range(len(features)):
            result_features.append(self.inputs[ii](features[ii]))
        
        f1, f2, f3, f4, f5, f6 = result_features

        f6_up = functional.interpolate(f6, scale_factor=2, mode='bilinear')
        f5_intermed = self.f5_intermeds[0](f6_up + f5)

        f5_up = functional.interpolate(f5_intermed, scale_factor=2, mode='bilinear')
        f4_intermed = self.f4_intermeds[0](f5_up + f4)

        f1_down = functional.max_pool2d(f1, kernel_size=3, stride=2, padding=1)
        f2_intermed = self.f2_intermed_1(f1_down + f2)

        f2_down = functional.max_pool2d(f2_intermed, kernel_size=3, stride=2, padding=1)
        f3_intermed = self.f3_intermeds[0](f2_down + f3)

        f4_up = functional.interpolate(f4_intermed, scale_factor=2, mode='bilinear')
        f3_intermed = self.f3_intermeds[1](f3_intermed + f4_up)

        f3_down = functional.max_pool2d(f3_intermed, kernel_size=3, stride=2, padding=1)
        f4_intermed = self.f4_intermeds[1](f3_down + f4_intermed + f4)

        f4_down = functional.max_pool2d(f4_intermed, kernel_size=3, stride=2, padding=1)
        f5_intermed = self.f5_intermeds[1](f4_down + f5_intermed + f5)
        
        f5_down = functional.max_pool2d(f5_intermed, kernel_size=3, stride=2, padding=1)
        f6_intermed = self.f6_intermed(f5_down + f6)

        f6_up = functional.interpolate(f6_intermed, scale_factor=2, mode='bilinear')
        f5_intermed = self.f5_intermeds[2](f6_up + f5_intermed + f5)

        f5_up = functional.interpolate(f5_intermed, scale_factor=2, mode='bilinear')
        f4_intermed = self.f4_intermeds[2](f5_up + f4_intermed + f4)

        f4_up = functional.interpolate(f4_intermed, scale_factor=2, mode='bilinear')
        f3_intermed = self.f3_intermeds[2](f4_up + f3_intermed + f3)

        f3_up = functional.interpolate(f3_intermed, scale_factor=2, mode='bilinear')
        f2_intermed = self.f2_intermed_2(f3_up + f2_intermed + f2)
        
        f2_up = functional.interpolate(f2_intermed, scale_factor=2, mode='bilinear')
        out = self.out(f2_up + f1)

        return out


@BACKBONES.register_module()
class SCMNet(BaseModule):
    """SCMNet backbone.

    This backbone is the implementation of `SCMNet: Shared Context Mining
    Network for Real-time Semantic Segmentation 
    <https://ieeexplore.ieee.org/abstract/document/9647401>`_.

    Args:
        #TODO
    """
    def __init__(self,
                in_channels=3,
                dsffm_intermed_channels=64,
                init_cfg=None,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
                dsffm_act_cfg=dict(type='Swish')):

        super(SCMNet, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # Inputs
        self.deep_in_conv = ConvModule(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(3,3),
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.shallow_in_conv = ConvModule(
            in_channels=in_channels,
            out_channels=24,
            kernel_size=(3,3),
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg, 
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # Before CMM 1
        self.deep_mbconv_1 = InvertedResidual(
            in_channels=32,
            out_channels=32,
            stride=2,
            expand_ratio=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.deep_mbconv_2 = InvertedResidual(
            in_channels=32,
            out_channels=48,
            stride=2,
            expand_ratio=6,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.deep_mbconv_3 = InvertedResidual(
            in_channels=48,
            out_channels=48,
            stride=1,
            expand_ratio=6,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.shallow_dsconv_1 = DepthwiseSeparableConvModule(
            in_channels=24,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.shallow_dsconv_2 = DepthwiseSeparableConvModule(
            in_channels=32,
            out_channels=48,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fuse_activation_1 = build_activation_layer(cfg=act_cfg)
        self.cmm_1 = ContextMiningModule(
            channels=48)

        # Before CMM 2
        self.deep_mbconv_4 = InvertedResidual(
            in_channels=48,
            out_channels=64,
            stride=2,
            expand_ratio=6,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.deep_mbconv_5 = InvertedResidual(
            in_channels=64,
            out_channels=64,
            stride=1,
            expand_ratio=6,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.deep_mbconv_6 = InvertedResidual(
            in_channels=64,
            out_channels=64,
            stride=1,
            expand_ratio=6,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.deep_mbconv_7 = InvertedResidual(
            in_channels=64,
            out_channels=96,
            stride=1,
            expand_ratio=6,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.shallow_dsconv_3 = DepthwiseSeparableConvModule(
            in_channels=48,
            out_channels=96,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fuse_activation_2 = build_activation_layer(cfg=act_cfg)
        self.cmm_2 = ContextMiningModule(
            channels=96)

        self.deep_mbconv_8 = InvertedResidual(
            in_channels=96,
            out_channels=128,
            stride=2,
            expand_ratio=6,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.deep_mbconv_9 = InvertedResidual(
            in_channels=128,
            out_channels=128,
            stride=1,
            expand_ratio=6,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.deep_mbconv_10 = InvertedResidual(
            in_channels=128,
            out_channels=128,
            stride=1,
            expand_ratio=6,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.deep_mbconv_11 = InvertedResidual(
            in_channels=128,
            out_channels=160,
            stride=1,
            expand_ratio=6,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.shallow_dsconv_4 = DepthwiseSeparableConvModule(
            in_channels=96,
            out_channels=160,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fuse_activation_3 = build_activation_layer(cfg=act_cfg)
        self.cmm_3 = ContextMiningModule(
            channels=160)

        self.maxpool = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=2,
            padding=1)

        self.dsffm = DeepShallowFeatureFusionModule(
            intermediate_channels=dsffm_intermed_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dsffm_act_cfg)
        

    def forward(self, x):
        x_deep = self.deep_in_conv(x)
        x_shallow_in = self.shallow_in_conv(x)

        x_deep = self.deep_mbconv_1(x_deep)
        x_deep = self.deep_mbconv_2(x_deep)
        x_deep = self.deep_mbconv_3(x_deep)

        x_shallow_1 = self.shallow_dsconv_1(x_shallow_in)
        x_shallow_2 = self.shallow_dsconv_2(x_shallow_1)

        x_fused = x_deep + x_shallow_2
        x_fused = self.fuse_activation_1(x_fused)
        x_cmm_1 = self.cmm_1(x_fused)

        x_deep = self.deep_mbconv_4(x_cmm_1)
        x_deep = self.deep_mbconv_5(x_deep)
        x_deep = self.deep_mbconv_6(x_deep)
        x_deep = self.deep_mbconv_7(x_deep)

        x_shallow = self.shallow_dsconv_3(x_fused)

        x_fused = x_deep + x_shallow
        x_fused = self.fuse_activation_2(x_fused)
        x_cmm_2 = self.cmm_2(x_fused)

        x_deep = self.deep_mbconv_8(x_fused)
        x_deep = self.deep_mbconv_9(x_deep)
        x_deep = self.deep_mbconv_10(x_deep)
        x_deep = self.deep_mbconv_11(x_deep)

        x_shallow = self.shallow_dsconv_4(x_fused)

        x_fused = x_deep + x_shallow
        x_fused = self.fuse_activation_3(x_fused)
        x_cmm_3 = self.cmm_3(x_fused)

        x_maxpool = self.maxpool(x_cmm_3)

        x = self.dsffm(
            x_shallow_in, 
            x_shallow_1, 
            x_cmm_1, 
            x_cmm_2, 
            x_cmm_3, 
            x_maxpool)

        # x = self.out_dsconv_1(x)
        # x = self.out_dsconv_2(x)

        x = tuple([x])

        return x