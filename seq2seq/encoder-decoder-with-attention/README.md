# Encoder-decoder model with Global Attention
## Overview
This is tiny implementation of Encoder-decoder model with Global Attention.
see [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

![](https://raw.githubusercontent.com/satopirka/nlp-nnabla/master/img/attention.png)

## Start training
This example demonstrates the training for English Japanese translation on small parallel corpus.

```sh
python train.py
```

If you want to use cuDNN, execute as follows

```
python train.py -c cudnn
```
