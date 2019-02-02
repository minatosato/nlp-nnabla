#
# Copyright (c) 2017-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import warnings
warnings.filterwarnings('ignore')
import sys
from gensim.models import KeyedVectors

w2v = KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)

query = sys.argv[1]
print(f'query = {query}')

for word, sim in w2v.most_similar(positive=[query]):
    print(f'{word}: {sim}')

