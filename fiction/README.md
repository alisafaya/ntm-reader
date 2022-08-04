

# Fiction/Non-fiction prediction results

Out of ~280K books in the Books3 corpus of Pile, the classifier labeled ~117K as fiction books. 

The precision of the classifier in detecting fiction books is around 98% on the test set. Detailed performance report of our classifier on the test set can be seen below:

```
Fiction vs Non-fiction:

               precision    recall  f1-score   support

           0      0.954     0.941     0.947       824
           1      0.980     0.985     0.982      2443

    accuracy                          0.974      3267
   macro avg      0.967     0.963     0.965      3267
weighted avg      0.974     0.974     0.974      3267
```

# Classification results for different categories

```
Christian:

               precision    recall  f1-score   support

           0      0.952     0.990     0.971       300
           1      0.944     0.773     0.850        66

    accuracy                          0.951       366
   macro avg      0.948     0.881     0.910       366
weighted avg      0.951     0.951     0.949       366


Religion & Spirituality:

               precision    recall  f1-score   support

           0      0.963     0.963     0.963       300
           1      0.963     0.963     0.963       300

    accuracy                          0.963       600
   macro avg      0.963     0.963     0.963       600
weighted avg      0.963     0.963     0.963       600


Poetry:

               precision    recall  f1-score   support

           0      0.987     1.000     0.993       300
           1      1.000     0.818     0.900        22

    accuracy                          0.988       322
   macro avg      0.993     0.909     0.947       322
weighted avg      0.988     0.988     0.987       322


Adventure:

               precision    recall  f1-score   support

           0      0.804     0.833     0.818       300
           1      0.747     0.708     0.727       209

    accuracy                          0.782       509
   macro avg      0.776     0.771     0.773       509
weighted avg      0.781     0.782     0.781       509


Adult:

               precision    recall  f1-score   support

           0      0.950     1.000     0.974       301
           1      1.000     0.111     0.200        18

    accuracy                          0.950       319
   macro avg      0.975     0.556     0.587       319
weighted avg      0.952     0.950     0.930       319


Horror:

               precision    recall  f1-score   support

           0      0.897     0.933     0.915       300
           1      0.747     0.648     0.694        91

    accuracy                          0.867       391
   macro avg      0.822     0.791     0.805       391
weighted avg      0.862     0.867     0.864       391

Historical:

               precision    recall  f1-score   support

           0      0.935     0.957     0.946       300
           1      0.867     0.810     0.837       105

    accuracy                          0.919       405
   macro avg      0.901     0.883     0.892       405
weighted avg      0.917     0.919     0.918       405


Romance:

               precision    recall  f1-score   support

           0      0.889     0.910     0.900       300
           1      0.908     0.887     0.897       300

    accuracy                          0.898       600
   macro avg      0.899     0.898     0.898       600
weighted avg      0.899     0.898     0.898       600


Fantasy:

               precision    recall  f1-score   support

           0      0.875     0.837     0.855       300
           1      0.843     0.880     0.861       300

    accuracy                          0.858       600
   macro avg      0.859     0.858     0.858       600
weighted avg      0.859     0.858     0.858       600


Science Fiction:

               precision    recall  f1-score   support

           0      0.852     0.883     0.867       300
           1      0.890     0.860     0.875       600

    accuracy                          0.871       600
   macro avg      0.871     0.872     0.871       600
weighted avg      0.872     0.871     0.871       600
```