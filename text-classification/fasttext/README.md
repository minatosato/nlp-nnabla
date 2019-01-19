# fastText
## Overview
This is implementation of fastText with bigram features.
see [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759).

## Start training
This example demonstrates the training for IMDB dataset.

```sh
python train.py
```

If you want to use cuDNN, execute as follows

```
python train.py -c cudnn
```

<img src="https://raw.githubusercontent.com/satopirka/nlp-nnabla/master/text-classification/fasttext/log/accuracy.png" style="width: 300px;">

<img src="https://raw.githubusercontent.com/satopirka/nlp-nnabla/master/text-classification/fasttext/log/cross-entropy.png" style="width: 300px;">