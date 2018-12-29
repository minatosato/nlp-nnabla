
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I
import numpy as np

from typing import Optional

def token_embedding(x: nn.Variable, vocab_size: int, embedding_size: int) -> nn.Variable:
    mask = get_mask(x)
    h = time_distributed(PF.embed)(x, vocab_size, embedding_size) * mask
    h *= embedding_size ** 0.5
    return h

def position_encoding(x: nn.Variable) -> nn.Variable:
    batch_size, sequence_length, dim = x.shape

    position = F.reshape(F.arange(0, sequence_length), shape=(sequence_length, 1))
    # -> (sequence_length, 1)
    div_term = F.exp(F.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    # -> (dim//2, )
    sin_val = F.sin(position * F.reshape(div_term, shape=(1, dim//2)))
    # -> (sequence_length, dim//2)
    cos_val = F.cos(position * F.reshape(div_term, shape=(1, dim//2)))
    # -> (sequence_length, dim//2)
    ret = []
    for i in range(dim):
        if i % 2 == 0:
            ret.append(sin_val[:, i//2:i//2+1])
        else:
            ret.append(cos_val[:, i//2:i//2+1])
    pe = F.reshape(F.concatenate(*ret, axis=1), shape=(1, sequence_length, dim))
    return x + F.broadcast(pe, shape=x.shape)

def layer_normalization(x: nn.Variable, eps:float=1e-6) -> nn.Variable:
    batch_size, sequence_length, dim = x.shape
    scale = nn.parameter.get_parameter_or_create('scale', shape=(1, 1, dim), initializer=I.ConstantInitializer(1.0))
    bias = nn.parameter.get_parameter_or_create('bias', shape=(1, 1, dim), initializer=I.ConstantInitializer(0.0))

    mean = F.mean(x, axis=2, keepdims=True)
    std = F.mean((x - mean)**2, axis=2, keepdims=True) ** 0.5
    return scale * (x - mean) / (std + eps) + bias


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


@PF.parametric_function_api('attention')
def attention(query, key, value, mask:Optional[nn.Variable]=None, train:bool=True, dropout_ratio:float=0.1, fix_parameters=False):
    '''
    A global attention layer
    Args:
        inputs (nnabla.Variable): A shape of [B, sen_len_query, units]
        memory (nnabla.Variable): A shape of [B, sen_len_memory, units]
        mask (nnabla.Variable): A shape of [B, sen_len_query, sen_len_memory]
        fix_parameters (bool): Fix parameters (Set need_grad=False).
    Returns:
        nn.Variable: A shape [B, units].
    '''
    batch_size, sentence_length_query, embedding_size =  query.shape
    batch_size, sentence_length_memory, embedding_size = key.shape
    q = query
    # -> (batch_size, sentence_length_query, embedding_size)
    k = key
    # -> (batch_size, sentence_length_memory, embedding_size)
    v = value
    # -> (batch_size, sentence_length_memory, embedding_size)
    

    logit = F.batch_matmul(q, k, transpose_b=True) * (embedding_size ** -0.5)
    # -> (batch_size, sentence_length_query, sentence_length_memory)
    
    # maskのshapeは-> (batch_size, sentence_length_query, sentence_length_memory)である
    if mask is not None:
        logit += get_attention_logit_mask(mask)


    attention_weights = F.softmax(logit, axis=2)
    # -> (batch_size, sentence_length_query, sentence_length_memory)

    if train:
        attention_weights = F.dropout(attention_weights, p=dropout_ratio)

    attention_output = F.batch_matmul(attention_weights, v)
    # -> (batch_size, sentence_length_query, embedding_size)

    return attention_output

def multihead_attention(query:nn.Variable, key:nn.Variable, value:nn.Variable, h:int, mask=None, train:bool=True, dropout_ratio:float=0.1):
    batch_size, sentence_length_query, embedding_size =  query.shape
    batch_size, sentence_length_memory, embedding_size = key.shape

    assert embedding_size % h == 0

    q = query
    k = key
    v = value

    dim = embedding_size // h

    with nn.parameter_scope('q_dense'):
        q = time_distributed(PF.affine)(q, embedding_size)
    with nn.parameter_scope('k_dense'):
        k = time_distributed(PF.affine)(k, embedding_size)
    with nn.parameter_scope('v_dense'):
        v = time_distributed(PF.affine)(v, embedding_size)

    q = F.reshape(q, shape=(batch_size, h, sentence_length_query, dim))
    k = F.reshape(k, shape=(batch_size, h, sentence_length_memory, dim))
    v = F.reshape(v, shape=(batch_size, h, sentence_length_memory, dim))

    ret = []
    # for h times
    for _q, _k, _v in zip(F.split(q, axis=1), F.split(k, axis=1), F.split(v, axis=1)):
        ret.append(attention(_q, _k, _v, mask=mask, train=train, dropout_ratio=dropout_ratio))

    x = F.concatenate(*ret, axis=2)
    with nn.parameter_scope('concat_dense'):
        x = time_distributed(PF.affine)(x, embedding_size)
    return x

def multihead_self_attention(x, h, mask=None, train:bool=True, dropout_ratio:float=0.1):
    return multihead_attention(x, x, x, h, mask=mask, train=train, dropout_ratio=dropout_ratio)

def positionwise_feed_forward(x, train:bool=True, dropout_ratio:float=0.1):
    batch_size, length, dim = x.shape
    with nn.parameter_scope('pff'):
        with nn.parameter_scope('w1'):
            h = F.relu(time_distributed(PF.affine)(x, dim*4))
        if train:
            h = F.dropout(h, p=dropout_ratio)
        with nn.parameter_scope('w2'):
            h = time_distributed(PF.affine)(h, dim)
    return h

def residual_normalization_wrapper(layer):
    def wrapper(x, *args, **kwargs):
        residual = x
        h = layer_normalization(x)
        h = layer(h, *args, **kwargs)
        if kwargs['train']:
            h = F.dropout(h, p=kwargs['dropout_ratio'])
        return residual + h
    return wrapper

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
    true_condition = F.reshape(condition, shape=list(condition.shape)+[1])
    print(true_condition)
    false_condition = F.constant(1, shape=true_condition.shape) - true_condition
    return true_condition * x + false_condition * y