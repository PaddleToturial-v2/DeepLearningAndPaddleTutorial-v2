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
import sys
import time
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

from config_utils import *
from model import AttentionCluster
from reader import FeatureReader
from metrics import get_metrics

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video test script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='AttentionCluster',
        help='name of model to test.')
    parser.add_argument(
        '--config',
        type=str,
        default='attention_cluster.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='test batch size. None to use config file setting.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--weights', type=str, default="./final", help="weight path")
    args = parser.parse_args()
    return args


def test(args):
    # parse config
    config = parse_config(args.config)
    test_config = merge_configs(config, 'test', vars(args))
    print_configs(test_config, 'Test')
    place = fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place):
        video_model = AttentionCluster("AttentionCluster", test_config, mode="test")

        model_dict, _ = fluid.load_dygraph(args.weights)
        video_model.set_dict(model_dict)

        test_reader = FeatureReader(name="ATTENTIONCLUSTER", mode='test', cfg=test_config)
        test_reader = test_reader.create_reader()

        video_model.eval()
        total_loss = 0.0
        total_acc1 = 0.0
        total_sample = 0

        for batch_id, data in enumerate(test_reader()):
            rgb = np.array([item[0] for item in data]).reshape([-1, 100, 1024]).astype('float32')
            audio = np.array([item[1] for item in data]).reshape([-1, 100, 128]).astype('float32')
            y_data = np.array([item[2] for item in data]).astype('float32')
            rgb = to_variable(rgb)
            audio = to_variable(audio)
            labels = to_variable(y_data)
            labels.stop_gradient = True
            output, logit = video_model([rgb,audio])
    
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=logit, label=labels)
            loss = fluid.layers.reduce_sum(loss, dim=-1)
            avg_loss = fluid.layers.mean(loss)
            # get metrics 
            valid_metrics = get_metrics(args.model_name.upper(), 'valid', test_config)
            hit_at_one,perr,gap = valid_metrics.calculate_and_log_out(loss, logit, labels, info = '[TEST] test_iter {} '.format(batch_id))
    
            total_loss += avg_loss.numpy()[0]
            total_acc1 += hit_at_one
            total_sample += 1
    
            print('TEST iter {}, loss = {}, acc1 {}'.format(
                batch_id,  avg_loss.numpy()[0], hit_at_one))
    
        print('Finish loss {} , acc1 {}'.format(
            total_loss / total_sample, total_acc1 / total_sample))


if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    test(args)
