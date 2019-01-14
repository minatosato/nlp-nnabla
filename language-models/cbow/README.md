## Countinuous Bag-of-Words model (CBOW)

### Usage

```
$ python cbow.py -c cudnn
```

After training, you can search the similar words for the query word.

```
$ python predict.py five
query = five
two: 0.47092053294181824
six: 0.4023873507976532
three: 0.39833539724349976
four: 0.3952694535255432
several: 0.3459046185016632
eight: 0.3441988229751587
coups: 0.3192448616027832
salt: 0.31761008501052856
would-be: 0.3053385615348816
westmoreland: 0.3019617199897766

$ python predict.py monday
query = monday
wednesday: 0.424960196018219
tuesday: 0.4056550860404968
september: 0.3816726803779602
friday: 0.3786037564277649
gorky: 0.3769057095050812
deeper: 0.3696543574333191
tomorrow: 0.3395729064941406
amoco: 0.3208698332309723
190-point: 0.3198375105857849
thursday: 0.3195176124572754
```
