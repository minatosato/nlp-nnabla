# 
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from collections import Counter
from typing import List
from typing import Tuple

def calc_sampling_prob(sentences: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    oneline: List[int] = []
    for sentence in sentences:
        oneline.extend(sentence)
    counter = Counter(oneline)
    words = np.array(list(counter.keys()), dtype=np.int32)
    prob = np.array(list(counter.values())) / sum(counter.values())
    prob = np.power(prob, 0.75)
    prob = prob / np.sum(prob)
    return words, prob

def negative_sampling(target: int, words: np.ndarray, prob: np.ndarray, k: int = 5):
    ret = np.random.choice(words, size=k, p=prob)
    while target in ret:
        ret = np.random.choice(words, size=k, p=prob)
    return ret

def to_cbow_dataset(sentences: List[List[int]], window_size: int = 1, ns: bool = False):
    contexts: List[List[int]] = []
    targets: List[int] = []

    if ns:
        negative_samples: List[np.ndarray] = []
        words, prob = calc_sampling_prob(sentences)

    for sentence in sentences:
        for _index in range(window_size, len(sentence)-window_size):
            targets.append(sentence[_index])
            if ns:
                negative_samples.append(negative_sampling(sentence[_index], words, prob))
            ctx: List[int] = []
            for t in range(-window_size, window_size+1):
                if t == 0:
                    continue
                ctx.append(sentence[_index + t])
            contexts.append(ctx)
    
    ret = [np.array(contexts, dtype=np.int32), np.array(targets, dtype=np.int32)[:, None]]
    if ns:
        ret.append(np.array(negative_samples, dtype=np.int32))
    return tuple(ret)

