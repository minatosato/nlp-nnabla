# Deep learning implementation for NLP with NNabla
Tiny implementation of deep learning models for NLP with Sony's NNabla.

## Test environment
- NNabla v0.9.7
- Keras v2.1.2 (for preprocessing)
- Numpy v1.13.3
- tqdm v4.19.5

## New features (different from the master respository of the NNabla)
- RNN layer
- LSTM layer
- Highway layer
- Time distributed parametric functions

## Models

### Language models
- A vanilla recurrent neural network language model (`rnnlm.py`)
- LSTM language model (`lstmlm.py`)
- Character-level neural language model (`char-cnn-lstmlm.py`)
  
  - this is not completely same implementation of a paper ["Character-Aware Neural Language Models"](https://arxiv.org/abs/1508.06615).

### Seq2Seq models
- Encoder-decoder (`encdec.py`)
- Encoder-decoder + global attention (`attention.py`)

## Future work
- Skip-gram model
- Continuous-BoW model
- Encoder-decoder + local attention
- Peephole LSTM
- GRU
- etc.
