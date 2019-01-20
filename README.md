# Deep learning implementation for NLP with NNabla
Tiny implementation of deep learning models for NLP with Sony's NNabla.

## Tested environment
- Python 3.7.2
- NNabla v1.0.10

## Models

### Language models / word embedding
- A vanilla recurrent neural network language model ([`language-models/rnnlm/`](https://github.com/satopirka/nlp-nnabla/tree/master/language-models/rnnlm))
- LSTM language model ([`language-models/lstmlm/`](https://github.com/satopirka/nlp-nnabla/blob/master/language-models/lstmlm))
- Character-level convolutional LSTM language model ([`language-models/char-cnn-lstmlm/`](https://github.com/satopirka/nlp-nnabla/blob/master/language-models/char-cnn-lstmlm))
- Continuous Bag-of-Words (CBOW) model ([`language-models/cbow/`](https://github.com/satopirka/nlp-nnabla/blob/master/language-models/cbow)))
- GloVe model ([`language-models/glove/`](https://github.com/satopirka/nlp-nnabla/blob/master/language-models/glove)))


### Seq2Seq models
- Encoder-decoder ([`seq2seq/encoder-decoder/`](https://github.com/satopirka/nlp-nnabla/blob/master/seq2seq/encoder-decoder))
- Encoder-decoder + global attention ([`seq2seq/encoder-decoder-with-attention/`](https://github.com/satopirka/nlp-nnabla/blob/master/seq2seq/encoder-decoder-with-attention))


### Text classifiers
- fastText ([`text-classifications/fasttext/`](https://github.com/satopirka/nlp-nnabla/blob/master/text-classification/fasttext))
- Self attention ([`text-classifications/self_attention/`](https://github.com/satopirka/nlp-nnabla/blob/master/text-classification/self-attention))
- LSTM classifier ([`text-classifications/lstm-classifier/`](https://github.com/satopirka/nlp-nnabla/blob/master/text-classification/self-attention))

## Future work
- Skip-gram model
- Peephole LSTM
- GRU
- Transformer
- ELMo
- etc.
