# A Structured Self-attentive Sentence Embedding
## Overview
This is implementation of self-attention network for text classification.
see [A Structured Self-attentive Sentence Embedding
](https://arxiv.org/abs/1703.03130).

![](https://user-images.githubusercontent.com/166852/33136258-ccc5bc08-cf72-11e7-8ddd-368e4a85a0a8.png)

## Start training
This example demonstrates the training for IMDB dataset.

```sh
python train.py
```

If you want to use cuDNN, execute as follows

```
python train.py -c cudnn
```

