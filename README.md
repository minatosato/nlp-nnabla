# Deep learning implementation for NLP with NNabla
Tiny implementation of deep learning models for NLP with Sony's NNabla.

## Test environment
- Python 3.6.7
- NNabla v1.0.10

## New functions (different from the NNabla v0.9.7)
### Parametric functions
- `simple_rnn`
- `lstm`
- `highway`

### Functions
- `time_distributed`
- `time_distributed_softmax_cross_entropy`

## Models

### Language models
- A vanilla recurrent neural network language model ([`rnnlm.py`](https://github.com/satopirka/nlp-nnabla/blob/master/language-models/rnnlm.py))
- LSTM language model ([`lstmlm.py`](https://github.com/satopirka/nlp-nnabla/blob/master/language-models/lstmlm.py))
- Character-level neural language model ([`char-cnn-lstmlm.py`](https://github.com/satopirka/nlp-nnabla/blob/master/language-models/char-cnn-lstmlm.py))
  
  - this is almost same implementation of the paper ["Character-Aware Neural Language Models"](https://arxiv.org/abs/1508.06615).

#### Usage

To start training of the model:

```bash
cd language-models
python char-cnn-lstm.py
```

If you can use cudnn,

```bash
python char-cnn-lstm.py -c cudnn
```

After training, you can get the similar words to the query word:

```python
In [3]: get_top_k('looooook', k=5)
Out[3]: ['look', 'looks', 'looked', 'loose', 'looking']

In [4]: get_top_k('while', k=5)
Out[4]: ['chile', 'whole', 'meanwhile', 'child', 'wholesale']

In [5]: get_top_k('richard', k=5)
Out[5]: ['richer', 'rich', 'michael', 'richter', 'richfield']
```

which is similar to the paper ["Character-Aware Neural Language Models"](https://arxiv.org/abs/1508.06615).

Pre-trained model is available ([here](https://github.com/satopirka/nlp-nnabla/releases/download/v0.0.1-beta/char-cnn-lstm_best.h5)).

> <img src="https://github.com/satopirka/nlp-nnabla/blob/master/img/char-cnn-lstm.png" style="width: 500px">

### Seq2Seq models
- Encoder-decoder ([`encdec.py`](https://github.com/satopirka/nlp-nnabla/blob/master/seq2seq/encdec.py))
- Encoder-decoder + global attention ([`attention.py`](https://github.com/satopirka/nlp-nnabla/blob/master/seq2seq/attention.py))

#### Usage

To start training of the model: 

```bash
cd seq2seq
./download.sh
ipython
run attention.py
```

And you can try to translate Japanese sentence into English by the model like below:

```python
nn.load_parameters('attention_en2ja.h5')

In [00]: translate("i was unable to look her in the face .")
Out[00]: '彼女の顔をまともに見ることが出来なかった。'

In [00]: translate("how far is it to the station ?")
Out[00]: '駅までどのくらいありますか。'
```

### Text classifiers
- fastText ([`fasttext.py`](https://github.com/satopirka/nlp-nnabla/blob/master/text-classification/fasttext.py))
- Self attention ([`self_attention.py`](https://github.com/satopirka/nlp-nnabla/blob/master/text-classification/self_attention.py))

## Future work
- Skip-gram model
- Continuous-BoW model
- Encoder-decoder + local attention
- Peephole LSTM
- GRU
- etc.
