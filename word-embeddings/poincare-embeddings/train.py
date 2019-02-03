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
from common.functions import expand_dims

from common.trainer import Trainer

from nnabla.utils.data_source_loader import download
from nnabla.utils.data_source_loader import get_data_home

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


"""
"""


embedding_size: int = 2
batch_size: int = 1
max_epoch: int = 100
negative_sample_size = 10

file_url = 'https://raw.githubusercontent.com/qiangsiwei/poincare_embedding/master/data/mammal_subtree.tsv'

from functools import reduce
import operator
import random

with download(file_url, open_file=True) as f:
    lines: str = f.read().decode('utf-8').split('\n')
    pdata = list(map(lambda l:l.split('\t'),filter(None,lines)))

pdict = {w:i for i,w in enumerate(set(reduce(operator.add, pdata)))}

vocab_size: int = len(pdict)
num_train_batch = len(pdata)//batch_size

def load_train_func(index):
    x, y = pdata[index]
    negative_sample_prob = np.ones(len(pdict))
    negative_sample_prob[pdict[x]] = 0.0
    negative_sample_prob[pdict[y]] = 0.0
    negative_sample_prob /= len(pdict) - 2
    negative_sample_indices = np.random.choice(range(len(pdict)), negative_sample_size, 
                                               replace=False, p=negative_sample_prob)
    return pdict[x], pdict[y], negative_sample_indices

train_data_iter = data_iterator_simple(load_train_func, len(pdata), batch_size, shuffle=True, with_file_cache=False)


"""
"""

def distance(u, v, eps=1e-5):
    uu = F.sum(F.pow_scalar(u, 2), axis=1)
    vv = F.sum(F.pow_scalar(v, 2), axis=1)
    euclid_norm_pow2 = F.sum(F.pow_scalar(u - v, 2), axis=1)
    alpha = F.maximum2(F.constant(eps, shape=uu.shape), 1.0 - uu)
    beta = F.maximum2(F.constant(eps, shape=vv.shape), 1.0 - vv)

    return F.acosh(1 + 2 * euclid_norm_pow2 / (alpha * beta))

		# alpha, beta = max(self.eps,1-uu), max(self.eps,1-vv)
		# gamma = max(1.,1+2*(uu-2*uv+vv)/alpha/beta)


def projection(x: nn.NdArray, eps: float = 1e-5) -> nn.NdArray:
    norm = F.pow_scalar(F.sum(x**2, axis=1), val=0.5)
    return F.where(condition=F.greater_equal_scalar(norm, val=1.),
                   x_true=F.clip_by_norm(x, clip_norm=1-eps, axis=1),
                   x_false=x)

    # return F.clip_by_norm(x, clip_norm=1-eps, axis=1)



class RiemannianSgd(S.Solver):
    def __init__(self, lr=0.01, eps=1e-5):
        self.lr = lr
        self.eps = eps
    
    def set_parameters(self, params: Dict[str, nn.Variable]):
        self.params = params

    def zero_grad(self):
        for key in self.params:
            self.params[key].grad.zero()

    def update(self):
        for key in self.params:
            rescaled_gradient: nn.NdArray = self.params[key].grad * (1. - F.sum(self.params[key].data**2, axis=1, keepdims=True))**2 / 4.
            # print(self.params[key].grad.data)
            # print(rescaled_gradient.data)
            if np.inf in self.params[key].grad.data:
                print(self.params[key].grad.data)
                exit()
            self.params[key].data -= self.lr * rescaled_gradient
            self.params[key].data = projection(self.params[key].data, eps=self.eps)

def loss_function(u, v, negative_samples):
    return F.sum(-F.log(F.exp(-distance(u, v)) / sum([F.exp(-distance(u, x)) for x in F.split(negative_samples, axis=2)])))

    # loss = -tf.log(tf.exp(-self.dists(u,v))/tf.reduce_sum(tf.exp(-self.dists(u,negs))))



u = nn.Variable((batch_size,))
v = nn.Variable((batch_size,))
negative_samples = nn.Variable((batch_size, negative_sample_size))

_u = PF.embed(u, vocab_size, embedding_size)
_v = PF.embed(v, vocab_size, embedding_size)
_neg = PF.embed(negative_samples, vocab_size, embedding_size)
_neg = F.transpose(_neg, axes=(0, 2, 1))

loss = loss_function(_u, _v, _neg)

nn.get_parameters()["embed/W"].d = I.UniformInitializer([-0.1, 0.1])(shape=(vocab_size, embedding_size))

solver = RiemannianSgd(lr=0.01)
solver.set_parameters(nn.get_parameters())

trainer = Trainer(inputs=[u, v, negative_samples], loss=loss, solver=solver)
trainer.run(train_data_iter, None, epochs=max_epoch)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,10))
ax = plt.gca()
ax.cla()
ax.set_xlim((-1.1,1.1)); ax.set_ylim((-1.1,1.1))
ax.add_artist(plt.Circle((0,0),1.,color='black',fill=False))
ax.grid()
for w,i in pdict.items():
    c0,c1 = nn.get_parameters()["embed/W"].d[i]
    ax.plot(c0,c1,'o',color='y')
    ax.text(c0+.01,c1+.01,w,color='b')
fig.savefig('./output.png',dpi=fig.dpi)

# In [2]: u.d
# Out[2]: array([14.], dtype=float32)

# In [3]: v.d
# Out[3]: array([20.], dtype=float32)

# In [4]: negative_samples.d
# Out[4]: array([[17.,  7., 23.,  3., 15.,  9., 12.,  5.,  1.,  6.]], dtype=float32)