# 
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np

from typing import Optional
from typing import Tuple

@PF.parametric_function_api('global_attention')
def global_attention(query: nn.Variable, memory: nn.Variable, mask: Optional[nn.Variable] = None,
                     score: str = 'general', fix_parameters: bool = False) -> nn.Variable:
    '''
    A global attention layer
    Args:
        query (nnabla.Variable): A shape of [batch_size, length_query, embedding_size]
        memory (nnabla.Variable): A shape of [batch_size, length_memory, embedding_size]
        mask (nnabla.Variable): A shape of [batch_size, length_query, length_memory]
        score (str): A kind of score functions for calculating attention weights.
                     'general', 'dot' or 'concat'.
                     see [Effective Approaches to Attention-based Neural Machine Translation]
                         (http://aclweb.org/anthology/D15-1166)
        fix_parameters (bool): Fix parameters (Set need_grad=False).
    Returns:
        nn.Variable: A shape [batch_size, length_query, embedding_size].
    '''
    batch_size, length_query, embedding_size =  query.shape
    _, length_memory, _ = memory.shape
    q = query
    # -> (batch_size, length_query, embedding_size)
    k = memory
    # -> (batch_size, length_memory, embedding_size)
    v = memory
    # -> (batch_size, length_memory, embedding_size)
    if score == 'dot':
        logit = F.batch_matmul(q, k, transpose_b=True)
        # -> (batch_size, length_query, length_memory)
    elif score == 'general':
        with nn.parameter_scope('Wa'):
            wa = time_distributed(PF.affine)(q, embedding_size, with_bias=False)
            # -> (batch_size, length_query, embeding_size)
        logit = F.batch_matmul(wa, k, transpose_b=True)
        # -> (batch_size, length_query, length_memory)
    elif score == 'concat':
        a_list = []
        for _q in F.split(q, axis=1):
            _q = F.reshape(_q, shape=(batch_size, 1, embedding_size))
            _q = F.broadcast(_q, shape=(batch_size, length_memory, embedding_size))
            concat = F.concatenate(_q, k, axis=2)
            # -> (batch_size, length_memory, embedding_size * 2)
            with nn.parameter_scope('Wa'):
                a = time_distributed(PF.affine)(concat, 1, with_bias=False)
                # -> (batch_size, length_memory, 1)
                a_list.append(a)
        
        logit = F.concatenate(*a_list, axis=2)
        # -> (batch_size, length_memory, length_query)
        logit = F.transpose(logit, axes=(0, 2, 1))
        # -> (batch_size, length_query, length_memory)
    
    # get_attention_logit_mask -> (batch_size, length_query, length_memory)である
    if mask is not None:
        logit += get_attention_logit_mask(mask)


    attention_weights = F.softmax(logit, axis=2)
    # -> (batch_size, length_query, length_memory)

    attention_output = F.batch_matmul(attention_weights, v)
    # -> (batch_size, length_query, embedding_size)

    return attention_output

def get_attention_logit_mask(mask: nn.Variable) -> nn.Variable:
    bit_inverted = F.constant(1, shape=mask.shape) - mask
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


