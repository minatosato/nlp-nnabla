# Poincaré Embeddings for Learning Hierarchical Representations
## Overview
This is tiny implementation of the paper [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/pdf/1705.08039.pdf).

## Start training
This example demonstrates the training for WordNet synset 'mammal.n.01'.

```
python train.py
```

If you want to use cuDNN, execute as follows

```
python train.py -c cudnn
```

![](https://raw.githubusercontent.com/satopirka/nlp-nnabla/master/word-embeddings/poincare-embeddings/output.png)


