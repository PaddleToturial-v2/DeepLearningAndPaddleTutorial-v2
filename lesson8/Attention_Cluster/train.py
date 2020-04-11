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
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='AttentionCluster',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='attention_cluster.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--no_memory_optimize',
        action='store_true',
        default=False,
        help='whether to use memory optimize in train')
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='validation epoch interval, 0 for no validation.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default=os.path.join('data', 'checkpoints'),
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--fix_random_seed',
        type=ast.literal_eval,
        default=False,
        help='If set True, enable continuous evaluation job.')
    args = parser.parse_args()
    return args


def val(epoch, model, cfg, args,valid_config):
    reader = FeatureReader(name="ATTENTIONCLUSTER", mode="valid", cfg=cfg)
    reader = reader.create_reader()
    total_loss = 0.0
    total_acc1 = 0.0
    total_sample = 0

    for batch_id, data in enumerate(reader()):
        rgb = np.array([item[0] for item in data]).reshape([-1, 100, 1024]).astype('float32')
        audio = np.array([item[1] for item in data]).reshape([-1, 100, 128]).astype('float32')
        y_data = np.array([item[2] for item in data]).astype('float32')
        rgb = to_variable(rgb)
        audio = to_variable(audio)
        labels = to_variable(y_data)
        labels.stop_gradient = True
        output, logit = model([rgb,audio])

        loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=logit, label=labels)
        loss = fluid.layers.reduce_sum(loss, dim=-1)
        avg_loss = fluid.layers.mean(loss)
        # get metrics 
        valid_metrics = get_metrics(args.model_name.upper(), 'valid', valid_config)
        hit_at_one,perr,gap = valid_metrics.calculate_and_log_out(loss, logit, labels, info = '[TEST] test_iter {} '.format(batch_id))

        total_loss += avg_loss.numpy()[0]
        total_acc1 += hit_at_one
        total_sample += 1

        print('TEST Epoch {}, iter {}, loss = {}, acc1 {}'.format(
            epoch, batch_id,
            avg_loss.numpy()[0], hit_at_one))

    print('Finish loss {} , acc1 {}'.format(
        total_loss / total_sample, total_acc1 / total_sample))


def create_optimizer(cfg, params):
    optimizer = fluid.optimizer.AdamOptimizer(cfg.learning_rate,
    parameter_list=params)

    return optimizer


def train(args):
    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))
    print_configs(train_config, 'Train')

    use_data_parallel = False
    trainer_count = fluid.dygraph.parallel.Env().nranks
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if use_data_parallel else fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place):
        if use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()

        video_model = AttentionCluster("AttentionCluster", train_config, mode="train")

        optimizer = create_optimizer(train_config.TRAIN,
                                     video_model.parameters())
        if use_data_parallel:
            video_model = fluid.dygraph.parallel.DataParallel(video_model,
                                                              strategy)

        bs_denominator = 1
        if args.use_gpu:
            # check number of GPUs
            gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
            if gpus == "":
                pass
            else:
                gpus = gpus.split(",")
                num_gpus = len(gpus)
                assert num_gpus == train_config.TRAIN.num_gpus, \
                       "num_gpus({}) set by CUDA_VISIBLE_DEVICES" \
                       "shoud be the same as that" \
                       "set in {}({})".format(
                       num_gpus, args.config, train_config.TRAIN.num_gpus)
            bs_denominator = train_config.TRAIN.num_gpus

        train_config.TRAIN.batch_size = int(train_config.TRAIN.batch_size /
                                            bs_denominator)

        train_reader = FeatureReader(name="ATTENTIONCLUSTER", mode="train", cfg=train_config)

        train_reader = train_reader.create_reader()
        if use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        for epoch in range(train_config.TRAIN.epoch):
            video_model.train()
            total_loss = 0.0
            total_acc1 = 0.0
            total_sample = 0
            for batch_id, data in enumerate(train_reader()):
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
                train_metrics = get_metrics(args.model_name.upper(), 'train', train_config)

                hit_at_one,perr,gap = train_metrics.calculate_and_log_out(loss, logit, labels, info = '[TRAIN] Epoch {}, iter {} '.format(epoch, batch_id))

                if use_data_parallel:
                    avg_loss = video_model.scale_loss(avg_loss)
                    avg_loss.backward()
                    video_model.apply_collective_grads()
                else:
                    avg_loss.backward()
                optimizer.minimize(avg_loss)
                video_model.clear_gradients()

                total_loss += avg_loss.numpy()[0]
                total_acc1 += hit_at_one
                total_sample += 1

                print('TRAIN Epoch {}, iter {}, loss = {}, acc1 {}'.
                      format(epoch, batch_id,
                             avg_loss.numpy()[0],
                             hit_at_one))

            print(
                'TRAIN End, Epoch {}, avg_loss= {}, avg_acc1= {}'.
                format(epoch, total_loss / total_sample, total_acc1 /total_sample))
            video_model.eval()
            val(epoch, video_model, valid_config, args, valid_config)

        if fluid.dygraph.parallel.Env().local_rank == 0:
            fluid.dygraph.save_dygraph(video_model.state_dict(), "final")
        logger.info('[TRAIN] training finished')



if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    train(args)
