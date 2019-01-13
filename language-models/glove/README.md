# GloVe: Global Vectors for Word Representation
This is tiny implementation of the paper [GloVe: Global Vectors for Word Representation](https://www.aclweb.org/anthology/D14-1162).

To start training, 

```
python glove.py -c cudnn
```

After training, you can search the similar words for the query word.

<img src="https://raw.githubusercontent.com/satopirka/nlp-nnabla/feature/embedding/language-models/glove/log/loss.png" style="width: 400px;">

```
$ python predict.py five
query = five
four: 0.809036374092102
six: 0.7296661138534546
three: 0.727622389793396
seven: 0.7273768782615662
eight: 0.7221855521202087
two: 0.6647053956985474
several: 0.6521549224853516
status: 0.6303826570510864
couple: 0.6245783567428589
subsidiaries: 0.618409276008606

$ python predict.py monday
query = monday
friday: 0.8147551417350769
tuesday: 0.7707927227020264
wednesday: 0.7579326629638672
thursday: 0.7538329362869263
quake: 0.6891617178916931
day: 0.6782408952713013
india: 0.6629637479782104
final: 0.6571143269538879
yesterday: 0.6518752574920654
technical: 0.6493551731109619
```
