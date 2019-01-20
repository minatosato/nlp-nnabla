# Character-Aware Neural Language Model
## Overview
This is implementation of Character-level Convolutional LSTM Language model. See the paper [Character-Aware Neural Language Model](https://arxiv.org/abs/1508.06615).

The network implemented in this code is composed of
- Character embedding layer
- Character-level convolutional network
- Highway network
- LSTM language model
.

![](https://raw.githubusercontent.com/carpedm20/lstm-char-cnn-tensorflow/master/assets/model.png)


## Start training
This example demonstrates the training for PTB dataset.

```
python train.py
```

If you want to use cuDNN, execute as follows

```
python train.py -c cudnn
```

