# Encoder-decoder model
## Overview
This is tiny implementation of Encoder-decoder model.
see [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)

![](https://raw.githubusercontent.com/satopirka/nlp-nnabla/master/img/seq2seq.png)

## Start training
This example demonstrates the training for English Japanese translation on small parallel corpus.

```sh
python train.py
```

If you want to use cuDNN, execute as follows

```
python train.py -c cudnn
```
