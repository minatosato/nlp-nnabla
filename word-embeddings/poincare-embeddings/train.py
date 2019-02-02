#
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
sys.path.append('../../')

import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I
import nnabla.solvers as S
import nnabla.monitor as M

from nnabla.utils.data_iterator import data_iterator_simple

from common.parametric_functions import lstm
from common.functions import time_distributed
from common.functions import time_distributed_softmax_cross_entropy
from common.functions import get_mask

from typing import Dict, Union

import argparse
parser = argparse.ArgumentParser(description='GloVe model training.')
parser.add_argument('--context', '-c', type=str,
                    default='cpu', help='You can choose cpu or cudnn.')
parser.add_argument('--device', '-d', type=int,
                    default=0, help='You can choose the device id when you use cudnn.')
args = parser.parse_args()

if args.context == 'cudnn':
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cudnn', device_id=args.device)
    nn.set_default_context(ctx)


def distance(u, v):
    uu = F.sum(F.pow_scalar(u, 2), axis=1)
    vv = F.sum(F.pow_scalar(u, 2), axis=1)
    euclid_norm_pow2 = F.sum(F.pow_scalar(u - v, 2), axis=1)
    return F.acosh(1 + 2 * euclid_norm_pow2 / ((1 - uu) * (1 - vv)))

def projection(x: nn.NdArray, eps: float = 1e-5) -> nn.NdArray:
    return F.clip_by_norm(x, clip_norm=1-eps, axis=1)


class RiemannianSgd(object):
    def __init__(self, lr=0.01, eps=1e-5):
        self.lr = lr
        self.eps = eps
    
    def set_parameters(self, params: Dict[str, nn.Variable]):
        self.params = params

    def zero_grad(self):
        for key in self.params:
            self.params[key].data.zero()
    
    def update(self):
        for key in self.params:
            rescaled_gradient: nn.NdArray = self.params[key].grad * (1. - F.sum(self.params[key].data**2, axis=1))**2 / 4.
            self.params[key].data -= self.lr * rescaled_gradient
            self.params[key].data = projection(self.params[key].data, eps=self.eps)

def loss_function(u, v, negative_samples):
    return F.sum(F.log(F.exp(-distance(u, v)) / sum([distance(u, x) for x in F.split(negative_samples, axis=2)])))


vocab_size: int = 10000
embedding_size: int = 5
batch_size: int = 128
max_epoch: int = 100
negative_sample_size = 5

u = nn.Variable((batch_size,))
v = nn.Variable((batch_size,))
negative_samples = nn.Variable((batch_size, negative_sample_size))

_u = PF.embed(u, vocab_size, embedding_size)
_v = PF.embed(v, vocab_size, embedding_size)
_neg = PF.embed(negative_samples, vocab_size, embedding_size)
_neg = F.transpose(_neg, axes=(0, 2, 1))

loss = loss_function(_u, _v, _neg)

nn.get_parameters()["embed/W"].d = I.UniformInitializer([-0.001, 0.001])(shape=(vocab_size, embedding_size))
