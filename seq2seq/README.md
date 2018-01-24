## Sequence-to-Sequence translation models
Tiny implementation of seq2seq translation models.

## Models

- Encoder-decoder (`encdec.py`)
- Encoder-decoder + Global Attention (`attention.py`)

## Requremenets
See [root of this repository](https://github.com/satopirka/nlp-nnabla).

## Usage

To start training of a model: 

```bash
./download.sh
ipython
run attention.py
```

You can use pre-trained attention model:

```bash
./download.sh
ipython
run attention.py
Ctrl+C (interrupt)

!wget hogehoge
nn.load_parameters('attention_en2ja.h5')
```

And you can try to translate by the model like below:

```ipython
nn.load_parameters('attention_en2ja.h5')

In [71]: translate("i was unable to look her in the face .")
Out[71]: '彼女の顔をまともに見ることが出来なかった。'

In [79]: translate("how far is it to the station ?")
Out[79]: '駅までどのくらいありますか。'

In [63]: translate("he is an very kind man .")
Out[63]: '彼はとても親切な男だ。'

In [64]: translate('she is a very kind woman .')
Out[64]: '彼女はとても親切な女性です。'

In [66]: translate("this is a pencil .")
Out[66]: 'これは鉛筆です。'

In [67]: translate("this is a pen .")
Out[67]: 'これはペンです。'
```