# 
# Copyright (c) 2017-2018 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

def get_mask(x: nn.Variable) -> nn.Variable:
    assert len(x.shape) == 2
    batch_size, max_len = x.shape
    mask = F.reshape(F.sign(x), shape=(batch_size, max_len, 1))
    return mask

def get_attention_logit_mask(mask: nn.Variable) -> nn.Variable:
    bit_inverted: nn.Variable = F.constant(1, shape=mask.shape) - mask
    # -> (batch_size, memory_length, 1)
    bit_inverted = F.transpose(bit_inverted, (0, 2, 1))
    # -> (batch_size, 1, memory_length)
    attention_mask = bit_inverted * F.constant(np.finfo(np.float32).min, shape=bit_inverted.shape)
    return attention_mask

def where(condition: nn.Variable, x:nn.Variable, y: nn.Variable) -> nn.Variable:
    '''
    This function returns x if condition is 1, and y if condition is 0.
    Args:
        condition (nnabla.Variable): A shape of (batch_size, 1)
        x (nnabla.Variable): A shape of (batch_size, embedding_size)
        y (nnabla.Variable): A shape of (batch_size, embedding_size)
    '''
    if x.ndim == 1:
        true_condition = F.reshape(condition, shape=list(condition.shape)+[1])
    else:
        true_condition = condition
    false_condition = F.constant(1, shape=true_condition.shape) - true_condition
    return true_condition * x + false_condition * y

def time_distributed(func):
    def time_distributed_func(x, *args, **kwargs):
        ret = []
        batch_size = x.shape[0]
        for x_ in F.split(x, axis=1):
            value = func(x_, *args, **kwargs)
            _, output_dim = value.shape
            ret.append(F.reshape(value, (batch_size, 1, output_dim)))
        return F.concatenate(*ret, axis=1)
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

    