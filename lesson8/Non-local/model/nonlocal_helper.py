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

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle
import paddle.fluid as fluid
from paddle.fluid import ParamAttr

from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear,Conv3D
# 3d spacetime nonlocal (v1, spatial downsample)

class spacetime_nonlocal(fluid.dygraph.Layer):
    def __init__(self, dim_in, dim_out, batch_size, prefix, dim_inner, cfg, \
                       test_mode = False, max_pool_stride = 2):
        super(spacetime_nonlocal, self).__init__()
        self.cfg = cfg
        self.prefix = prefix
        self.dim_inner = dim_inner
        self.max_pool_stride = max_pool_stride
        self.conv3d_1 =  Conv3D(
                num_channels=dim_in,
                num_filters=dim_inner,
                filter_size=1,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(loc=0.0, scale=cfg.NONLOCAL.conv_init_std)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.)))
        
        self.conv3d_2 = Conv3D(
                num_channels=dim_in,
                num_filters=dim_inner,
                filter_size=1,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(loc=0.0, scale=cfg.NONLOCAL.conv_init_std)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.)))
    
        self.conv3d_3 = Conv3D(
                    num_channels=dim_in,
                    num_filters=dim_inner,
                    filter_size=1,
                    param_attr=ParamAttr(initializer=fluid.initializer.Normal(loc=0.0, scale=cfg.NONLOCAL.conv_init_std)),
                    bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.)))
    
        self.conv3d_4 = Conv3D(
                num_channels=dim_inner,
                num_filters=dim_out,
                filter_size=1,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(loc=0.0, scale=cfg.NONLOCAL.conv_init_std)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.)))
        
        self.bn = BatchNorm(
                num_channels=dim_out,
                is_test=test_mode,
                momentum=cfg.NONLOCAL.bn_momentum,
                epsilon=cfg.NONLOCAL.bn_epsilon,
                param_attr=ParamAttr(
                    initializer=fluid.initializer.Constant(
                        value=cfg.NONLOCAL.bn_init_gamma),
                    regularizer=fluid.regularizer.L2Decay(
                        cfg.TRAIN.weight_decay_bn)),
                bias_attr=ParamAttr(
                    regularizer=fluid.regularizer.L2Decay(
                        cfg.TRAIN.weight_decay_bn)))
    
    def forward(self, blob_in):
        cur = blob_in
        theta = self.conv3d_1(cur)
        theta_shape = theta.shape

        if self.cfg.NONLOCAL.use_maxpool:
            max_pool = fluid.layers.pool3d(
                input=cur,
                pool_size=[1, self.max_pool_stride, self.max_pool_stride],
                pool_type='max',
                pool_stride=[1, self.max_pool_stride, self.max_pool_stride],
                pool_padding=[0, 0, 0],
                name=self.prefix + '_pool')
        else:
            max_pool = cur
    

        phi = self.conv3d_2(max_pool)
        phi_shape = phi.shape

        g = self.conv3d_3(max_pool)
        g_shape = g.shape

        # we have to use explicit batch size (to support arbitrary spacetime size)
        # e.g. (8, 1024, 4, 14, 14) => (8, 1024, 784)
        theta = fluid.layers.reshape(
            theta, [-1, 0, theta_shape[2] * theta_shape[3] * theta_shape[4]])
        theta = fluid.layers.transpose(theta, [0, 2, 1])
        phi = fluid.layers.reshape(
            phi, [-1, 0, phi_shape[2] * phi_shape[3] * phi_shape[4]])
        theta_phi = fluid.layers.matmul(theta, phi, name=self.prefix + '_affinity')
        g = fluid.layers.reshape(g, [-1, 0, g_shape[2] * g_shape[3] * g_shape[4]])
        if self.cfg.NONLOCAL.use_softmax:
            if self.cfg.NONLOCAL.use_scale is True:
                theta_phi_sc = fluid.layers.scale(theta_phi, scale=self.dim_inner**-.5)
            else:
                theta_phi_sc = theta_phi
            p = fluid.layers.softmax(
                theta_phi_sc, name=self.prefix + '_affinity' + '_prob')
        else:
            # not clear about what is doing in xlw's code
            p = None  # not implemented
            raise "Not implemented when not use softmax"
    
        # note g's axis[2] corresponds to p's axis[2]
        # e.g. g(8, 1024, 784_2) * p(8, 784_1, 784_2) => (8, 1024, 784_1)
        p = fluid.layers.transpose(p, [0, 2, 1])
        t = fluid.layers.matmul(g, p, name=self.prefix + '_y')
    
        # reshape back
        # e.g. (8, 1024, 784) => (8, 1024, 4, 14, 14)
        t_shape = t.shape
        # print(t_shape)
        # print(theta_shape)
        t_re = fluid.layers.reshape(t, shape=list(theta_shape))
        blob_out = t_re
    
        blob_out = self.conv3d_4(blob_out)
        blob_out_shape = blob_out.shape

        if self.cfg.NONLOCAL.use_bn is True:
            bn_name = self.prefix + "_bn"
            blob_out = self.bn(blob_out)  # add bn
    
        if self.cfg.NONLOCAL.use_affine is True:
            affine_scale = fluid.layers.create_parameter(
                shape=[blob_out_shape[1]],
                dtype=blob_out.dtype,
                attr=ParamAttr(name=self.prefix + '_affine' + '_s'),
                default_initializer=fluid.initializer.Constant(value=1.))
            affine_bias = fluid.layers.create_parameter(
                shape=[blob_out_shape[1]],
                dtype=blob_out.dtype,
                attr=ParamAttr(name=self.prefix + '_affine' + '_b'),
                default_initializer=fluid.initializer.Constant(value=0.))
            blob_out = fluid.layers.affine_channel(
                blob_out,
                scale=affine_scale,
                bias=affine_bias,
                name=self.prefix + '_affine')  # add affine

        return blob_out


def add_nonlocal(blob_in,
                 dim_in,
                 dim_out,
                 batch_size,
                 prefix,
                 dim_inner,
                 cfg,
                 test_mode=False):
    net = spacetime_nonlocal(
                dim_in, dim_out, batch_size, prefix, dim_inner, cfg, test_mode = test_mode)
    blob_out = net(blob_in)
    blob_out = fluid.layers.elementwise_add(
        blob_out, blob_in, name=prefix + '_sum')
    return blob_out


# this is to reduce memory usage if the feature maps are big
# devide the feature maps into groups in the temporal dimension,
# and perform non-local operations inside each group.
def add_nonlocal_group(blob_in,
                       dim_in,
                       dim_out,
                       batch_size,
                       pool_stride,
                       height,
                       width,
                       group_size,
                       prefix,
                       dim_inner,
                       cfg,
                       test_mode=False):
    group_num = int(pool_stride / group_size)
    assert (pool_stride % group_size == 0), \
           'nonlocal block {}: pool_stride({}) should be divided by group size({})'.format(prefix, pool_stride, group_size)

    if group_num > 1:
        blob_in = fluid.layers.transpose(
            blob_in, [0, 2, 1, 3, 4], name=prefix + '_pre_trans1')
        blob_in = fluid.layers.reshape(
            blob_in,
            [batch_size * group_num, group_size, dim_in, height, width],
            name=prefix + '_pre_reshape1')
        blob_in = fluid.layers.transpose(
            blob_in, [0, 2, 1, 3, 4], name=prefix + '_pre_trans2')


    net = spacetime_nonlocal(
                dim_in, dim_out, batch_size, prefix, dim_inner, cfg, test_mode = test_mode)
    blob_out = net(blob_in)
    
    blob_out = fluid.layers.elementwise_add(
        blob_out, blob_in, name=prefix + '_sum')

    if group_num > 1:
        blob_out = fluid.layers.transpose(
            blob_out, [0, 2, 1, 3, 4], name=prefix + '_post_trans1')
        blob_out = fluid.layers.reshape(
            blob_out,
            [batch_size, group_num * group_size, dim_out, height, width],
            name=prefix + '_post_reshape1')
        blob_out = fluid.layers.transpose(
            blob_out, [0, 2, 1, 3, 4], name=prefix + '_post_trans2')

    return blob_out
