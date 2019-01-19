# LSTM text classifier
## Overview
This is implementation of simple LSTM network for text classification.

![](https://www.researchgate.net/profile/Huy_Tien_Nguyen/publication/321259272/figure/fig2/AS:572716866433034@1513557749934/Illustration-of-our-LSTM-model-for-sentiment-classification-Each-word-is-transfered-to-a.png)

## Start training
This example demonstrates the training for IMDB dataset.

```sh
python train.py
```

If you want to use cuDNN, execute as follows

```
python train.py -c cudnn
```

