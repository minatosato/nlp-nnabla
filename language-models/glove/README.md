# GloVe: Global Vectors for Word Representation
## Overview
This is tiny implementation of the paper [GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162).

## Start training
This example demonstrates the training for PTB dataset.

```
python train.py
```

If you want to use cuDNN, execute as follows

```
python train.py -c cudnn
```

<img src="https://raw.githubusercontent.com/satopirka/nlp-nnabla/feature/embedding/language-models/glove/log/loss.png" style="width: 400px;">


## Note
After training, you can search the similar words for the query word.

```
pip install -r requirements.txt

python predict.py monday
query = monday
friday: 0.6632016897201538
tuesday: 0.6016873717308044
thursday: 0.5354608297348022
wednesday: 0.4639507532119751
last: 0.4476850926876068
according: 0.4421117901802063
late: 0.4383971393108368
oct.: 0.4049926698207855
hot: 0.39583930373191833
evidence: 0.39306509494781494

$ python predict.py i
query = i
we: 0.573616623878479
you: 0.5637986063957214
there: 0.5546776652336121
can: 0.5442103743553162
he: 0.5205755233764648
my: 0.5184576511383057
she: 0.4964057505130768
do: 0.4944441318511963
did: 0.4907638132572174
what: 0.4785623848438263
```
