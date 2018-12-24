# 
# Copyright (c) 2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

def frobenius(x):
    return F.mean(F.sum(F.sum(x ** 2, axis=2), axis=1) ** 0.5)

def batch_eye(batch_size, size):
    return F.broadcast(F.reshape(F.matrix_diag(F.constant(1, shape=(size,))), shape=(1, size, size)), shape=(batch_size, size, size))

def time_distributed(func):
    def time_distributed_func(x, *args, **kwargs):
        ret = []
        batch_size = x.shape[0]
        length = x.shape[1]
        dim = x.shape[2] if x.ndim > 2 else 1
        if length > 1:
            xs = F.split(x, axis=1)
        else:
            xs = [F.reshape(x, (batch_size, dim))]
        for x_ in xs:
            value = func(x_, *args, **kwargs)
            _, output_dim = value.shape
            ret.append(F.reshape(value, (batch_size, 1, output_dim)))
        
        if length > 1:
            return F.concatenate(*ret, axis=1)
        else:
            return ret[0]
    return time_distributed_func


def time_distributed_softmax_cross_entropy(y_pred, y_true):
    '''
    A time distributed softmax crossentropy
    Args:
        y_pred (nnabla.Variable): A shape of [B, SentenceLength, O]. # one-hot
        y_true (nnabla.Variable): A shape of [B, SentenceLength, 1]. # index
    Returns:
        nn.Variable: A shape [B, SentenceLength].
    '''
    ret = []
    for y_p, y_t in zip(F.split(y_pred, axis=1), F.split(y_true, axis=1)):
        ret.append(F.softmax_cross_entropy(y_p, y_t))
    return F.concatenate(*ret)

    