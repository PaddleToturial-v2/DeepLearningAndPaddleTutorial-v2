#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import time
import sys
import paddle.fluid as fluid
from paddle.fluid import ParamAttr
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear,Conv3D
from model import resnet_video
from model import resnet_helper
import math
import numpy as np

import logging
logger = logging.getLogger(__name__)
__all__ = ["NonLocal"]

BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
}


class NonLocal(fluid.dygraph.Layer):
    def __init__(self, name, cfg, mode='train'):
        super(NonLocal, self).__init__()
        self.name = name
        self.cfg = cfg
        self.mode = mode
        self.is_training = (mode == 'train')
        self.linear = Linear(10,10)
        self.get_config()
        
        self.use_temp_convs_set, self.temp_strides_set, self. pool_stride = resnet_video.obtain_arc(cfg.MODEL.video_arc_choice, cfg[mode.upper()]['video_length'])
        self.conv3d = Conv3D(
            num_channels=3,
            num_filters=64,
            filter_size=[1 + self.use_temp_convs_set[0][0] * 2, 7, 7],
            stride=[self.temp_strides_set[0][0], 2, 2],
            padding=[self.use_temp_convs_set[0][0], 3, 3],
            param_attr=ParamAttr(initializer=fluid.initializer.MSRA()),
            bias_attr=False)
            
        self.test_mode = False if (mode == 'train') else True
        self.bn_conv1 = BatchNorm(num_channels=64,
            is_test=self.test_mode,
            momentum=cfg.MODEL.bn_momentum,
            epsilon=cfg.MODEL.bn_epsilon,
            param_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(
                    cfg.TRAIN.weight_decay_bn)),
            bias_attr=ParamAttr(
                regularizer=fluid.regularizer.L2Decay(
                    cfg.TRAIN.weight_decay_bn)),
            moving_mean_name= "bn_conv1_mean",
            moving_variance_name="bn_conv1_variance")

        
        self.fc = Linear(2048, cfg.MODEL.num_classes,
                param_attr=ParamAttr(
                    initializer=fluid.initializer.Normal(loc=0.0, scale=cfg.MODEL.fc_init_std)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.)))
    def get_config_from_sec(self, sec, item, default=None):
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)
        
    def get_config(self):
        # video_length
        self.video_length = self.get_config_from_sec(self.mode, 'video_length')
        # crop size
        self.crop_size = self.get_config_from_sec(self.mode, 'crop_size')
     
    def forward(self, inputs, cfg):
        conv_blob = self.conv3d(inputs)
        bn_blob = self.bn_conv1(conv_blob)
        relu_blob = fluid.layers.relu(bn_blob, name='res_conv1_bn_relu')
        # max pool
        max_pool = fluid.layers.pool3d(
            input=relu_blob,
            pool_size=[1, 3, 3],
            pool_type='max',
            pool_stride=[1, 2, 2],
            pool_padding=[0, 0, 0],
            name='pool1')
        # building res block
        if cfg.MODEL.depth in [50, 101]:
            group = cfg.RESNETS.num_groups
            width_per_group = cfg.RESNETS.width_per_group
            batch_size = int(cfg.TRAIN.batch_size / cfg.TRAIN.num_gpus)
            res_block = resnet_helper._generic_residual_block_3d
            dim_inner = group * width_per_group
            (n1, n2, n3, n4) = BLOCK_CONFIG[cfg.MODEL.depth]
            blob_in, dim_in = resnet_helper.res_stage_nonlocal(
                res_block,
                max_pool,
                64,
                256,
                stride=1,
                num_blocks=n1,
                prefix='res2',
                cfg=cfg,
                dim_inner=dim_inner,
                group=group,
                use_temp_convs=self.use_temp_convs_set[1],
                temp_strides=self.temp_strides_set[1],
                test_mode=self.test_mode)
    
            layer_mod = cfg.NONLOCAL.layer_mod
            if cfg.MODEL.depth == 101:
                layer_mod = 2
            if cfg.NONLOCAL.conv3_nonlocal is False:
                layer_mod = 1000
    
            blob_in = fluid.layers.pool3d(
                blob_in,
                pool_size=[2, 1, 1],
                pool_type='max',
                pool_stride=[2, 1, 1],
                pool_padding=[0, 0, 0],
                name='pool2')
    
            if cfg.MODEL.use_affine is False:
                blob_in, dim_in = resnet_helper.res_stage_nonlocal(
                    res_block,
                    blob_in,
                    dim_in,
                    512,
                    stride=2,
                    num_blocks=n2,
                    prefix='res3',
                    cfg=cfg,
                    dim_inner=dim_inner * 2,
                    group=group,
                    use_temp_convs=self.use_temp_convs_set[2],
                    temp_strides=self.temp_strides_set[2],
                    batch_size=batch_size,
                    nonlocal_name="nonlocal_conv3",
                    nonlocal_mod=layer_mod,
                    test_mode=self.test_mode)
            else:
                crop_size = cfg[mode.upper()]['crop_size']
                blob_in, dim_in = resnet_helper.res_stage_nonlocal_group(
                    res_block,
                    blob_in,
                    dim_in,
                    512,
                    stride=2,
                    num_blocks=n2,
                    prefix='res3',
                    cfg=cfg,
                    dim_inner=dim_inner * 2,
                    group=group,
                    use_temp_convs=self.use_temp_convs_set[2],
                    temp_strides=self.temp_strides_set[2],
                    batch_size=batch_size,
                    pool_stride=self.pool_stride,
                    spatial_dim=int(crop_size / 8),
                    group_size=4,
                    nonlocal_name="nonlocal_conv3_group",
                    nonlocal_mod=layer_mod,
                    test_mode=self.test_mode)
    
            layer_mod = cfg.NONLOCAL.layer_mod
            if cfg.MODEL.depth == 101:
                layer_mod = layer_mod * 4 - 1
            if cfg.NONLOCAL.conv4_nonlocal is False:
                layer_mod = 1000
    
            blob_in, dim_in = resnet_helper.res_stage_nonlocal(
                res_block,
                blob_in,
                dim_in,
                1024,
                stride=2,
                num_blocks=n3,
                prefix='res4',
                cfg=cfg,
                dim_inner=dim_inner * 4,
                group=group,
                use_temp_convs=self.use_temp_convs_set[3],
                temp_strides=self.temp_strides_set[3],
                batch_size=batch_size,
                nonlocal_name="nonlocal_conv4",
                nonlocal_mod=layer_mod,
                test_mode=self.test_mode)
    
            blob_in, dim_in = resnet_helper.res_stage_nonlocal(
                res_block,
                blob_in,
                dim_in,
                2048,
                stride=2,
                num_blocks=n4,
                prefix='res5',
                cfg=cfg,
                dim_inner=dim_inner * 8,
                group=group,
                use_temp_convs=self.use_temp_convs_set[4],
                temp_strides=self.temp_strides_set[4],
                test_mode=self.test_mode)
    
        else:
            raise Exception("Unsupported network settings.")
    
        blob_out = fluid.layers.pool3d(
            blob_in,
            pool_size=[self.pool_stride, 7, 7],
            pool_type='avg',
            pool_stride=[1, 1, 1],
            pool_padding=[0, 0, 0],
            name='pool5')
    
        if (cfg.TRAIN.dropout_rate > 0):
            blob_out = fluid.layers.dropout(
                blob_out, cfg.TRAIN.dropout_rate, is_test=self.test_mode)
    
        blob_out = fluid.layers.squeeze(input=blob_out,axes = [2,3,4])
        blob_out = self.fc(blob_out)
    
        softmax = fluid.layers.softmax(blob_out)

        return softmax

def get_learning_rate_decay_list(base_learning_rate, lr_decay, step_lists):
    lr_bounds = []
    lr_values = [base_learning_rate * 1]
    cur_step = 0
    for i in range(len(step_lists)):
        cur_step += step_lists[i]
        lr_bounds.append(cur_step)
        decay_rate = lr_decay**(i + 1)
        lr_values.append(base_learning_rate * decay_rate)

    return lr_bounds, lr_values