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
from paddle.fluid.dygraph.nn import Conv2D, Linear
import math
import numpy as np

__all__ = ["AttentionCluster"]


class ShiftingAttentionModel(object):
    """Shifting Attention Model"""

    def __init__(self, input_dim, seg_num, n_att, name):
        super(ShiftingAttentionModel, self).__init__()
        self.n_att = n_att
        self.input_dim = input_dim
        self.seg_num = seg_num
        self.name = name
        self.gnorm = np.sqrt(n_att)
        self.conv = Conv2D(
            num_channels=self.input_dim,
            num_filters=n_att,
            filter_size=1,
            param_attr=ParamAttr(
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.MSRA()))

    def softmax_m1(self, x):
        x_shape = fluid.layers.shape(x)
        x_shape.stop_gradient = True
        flat_x = fluid.layers.reshape(x, shape=(-1, self.seg_num))
        flat_softmax = fluid.layers.softmax(flat_x)
        return fluid.layers.reshape(
            flat_softmax, shape=x.shape, actual_shape=x_shape)

    def glorot(self, n):
        return np.sqrt(1.0 / np.sqrt(n))

    def forward(self, x):
        """Forward shifting attention model.

        Args:
          x: input features in shape of [N, L, F].

        Returns:
          out: output features in shape of [N, F * C]
        """

        trans_x = fluid.layers.transpose(x, perm=[0, 2, 1])
        # scores and weight in shape [N, C, L], sum(weights, -1) = 1
        trans_x = fluid.layers.unsqueeze(trans_x, [-1])
        # trans_x = to_variable(trans_x)
        scores = self.conv(trans_x)
        # print(scores)
        scores = fluid.layers.squeeze(scores, [-1])
        weights = self.softmax_m1(scores)

        glrt = self.glorot(self.n_att)
        self.w = fluid.layers.create_parameter(
            shape=(self.n_att, ),
            dtype=x.dtype,
            default_initializer=fluid.initializer.Normal(0.0, glrt))
        self.b = fluid.layers.create_parameter(
            shape=(self.n_att, ),
            dtype=x.dtype,
            default_initializer=fluid.initializer.Normal(0.0, glrt))

        outs = []
        for i in range(self.n_att):
            # slice weight and expand to shape [N, L, C]
            weight = fluid.layers.slice(
                weights, axes=[1], starts=[i], ends=[i + 1])
            weight = fluid.layers.transpose(weight, perm=[0, 2, 1])
            weight = fluid.layers.expand(weight, [1, 1, self.input_dim])

            w_i = fluid.layers.slice(self.w, axes=[0], starts=[i], ends=[i + 1])
            b_i = fluid.layers.slice(self.b, axes=[0], starts=[i], ends=[i + 1])
            shift = fluid.layers.reduce_sum(x * weight, dim=1) * w_i + b_i

            l2_norm = fluid.layers.l2_normalize(shift, axis=-1)
            outs.append(l2_norm / self.gnorm)

        out = fluid.layers.concat(outs, axis=1)
        return out

class LogisticModel(object):
    """Logistic model."""
    """Creates a logistic model.

    Args:
    model_input: 'batch' x 'num_features' matrix of input features.
    vocab_size: The number of classes in the dataset.

    Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    batch_size x num_classes."""
    
    def __init__(self, vocab_size):
        super(LogisticModel, self).__init__()
        self.logit = Linear(
            input_dim=4096,
            output_dim=vocab_size,
            act=None,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.MSRAInitializer(uniform=False)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.MSRAInitializer(uniform=False)))


    def forward(self, model_input):
        logit = self.logit(model_input)
        output = fluid.layers.sigmoid(logit)
        return output, logit


class AttentionCluster(fluid.dygraph.Layer):
    def __init__(self, name, cfg, mode='train'):
        super(AttentionCluster, self).__init__()
        self.name = name
        self.cfg = cfg
        self.mode = mode
        self.is_training = (mode == 'train')
        self.get_config()


        self.fc1 = Linear(
            input_dim=36864,
            output_dim=1024,
            act='tanh',
            param_attr=ParamAttr(
                name="fc1.weights",
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name="fc1.bias", initializer=fluid.initializer.MSRA()))
        self.fc2 = Linear(
            input_dim=1024,
            output_dim=4096,
            act='tanh',
            param_attr=ParamAttr(
                name="fc2.weights",
                initializer=fluid.initializer.MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name="fc2.bias", initializer=fluid.initializer.MSRA()))
    
    def get_config(self):
        # get model configs
        self.feature_num = self.cfg.MODEL.feature_num
        self.feature_names = self.cfg.MODEL.feature_names
        self.feature_dims = self.cfg.MODEL.feature_dims
        self.cluster_nums = self.cfg.MODEL.cluster_nums
        self.seg_num = self.cfg.MODEL.seg_num
        self.class_num = self.cfg.MODEL.num_classes
        self.drop_rate = self.cfg.MODEL.drop_rate

        if self.mode == 'train':
            self.learning_rate = self.get_config_from_sec('train',
                                                          'learning_rate', 1e-3)

    def get_config_from_sec(self, sec, item, default=None):
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)

    def forward(self, inputs):
        att_outs = []
        for i, (input_dim, cluster_num, feature) in enumerate(
                zip(self.feature_dims, self.cluster_nums, inputs)):
            att = ShiftingAttentionModel(input_dim, self.seg_num, cluster_num,
                                         "satt{}".format(i))
            att_out = att.forward(feature)
            att_outs.append(att_out)
        out = fluid.layers.concat(att_outs, axis=1)
        if self.drop_rate > 0.:
            out = fluid.layers.dropout(
                out, self.drop_rate, is_test=(not self.is_training))
        
        out = self.fc1(out)
        out = self.fc2(out)
        
        aggregate_model = LogisticModel(vocab_size=self.class_num)
        output, logit = aggregate_model.forward(out)

        return output, logit