Here are some models that can be used for hate speech detection.<br> 
The task is carried out by training the models on an annotated data present in the data as train.tsv<br>
These models can then be used to classify some unknown tweets.<br>
A classification of the a collection of tweets is done and present in the predictions folder.<br> 
But any collection of tweets can be used just replace the test.tsv in data folder with your test data.<br>

On running the main.py file, one can see the metrics for different models and can decide which model to choose for the classification.<br>

Here four classification models are used<br>
1. RandomForestClassifier With Tweets converted to tfidf vectors
2. SVM classifier and using pretrained embeddings of the spacy (en_core_web_md)
3. Fasttext classifier
4. A simple SGD Classifier trained after balancing the data

For using, just clone the repo and then run ./setup.sh in the directory of the repo.<br>
This will install the required files.<br>

Annotations: 0 indicates normal tweets and 1 indicates hateful speech in the train data used

### Results <br> 
```
metrics for RF
              precision    recall  f1-score   support

           0       0.83      0.92      0.87      2121
           1       0.81      0.64      0.71      1102

    accuracy                           0.83      3223
   macro avg       0.82      0.78      0.79      3223
weighted avg       0.82      0.83      0.82      3223
```
```
metrics for SVM
              precision    recall  f1-score   support

           0       0.82      0.89      0.86      2121
           1       0.75      0.63      0.68      1102

    accuracy                           0.80      3223
   macro avg       0.79      0.76      0.77      3223
weighted avg       0.80      0.80      0.80      3223
```
```
metrics for FT
Read 0M words
Number of words:  21556
Number of labels: 2
Progress: 100.0% words/sec/thread: 1619653 lr:  0.000000 avg.loss:  0.382171 ETA:   0h 0m 0s
              precision    recall  f1-score   support

           0       0.83      0.90      0.86      2121
           1       0.77      0.64      0.70      1102

    accuracy                           0.81      3223
   macro avg       0.80      0.77      0.78      3223
weighted avg       0.81      0.81      0.81      3223
```
```
metrics for SGD
              precision    recall  f1-score   support

           0       0.88      0.85      0.86      2121
           1       0.73      0.77      0.75      1102

    accuracy                           0.82      3223
   macro avg       0.80      0.81      0.81      3223
weighted avg       0.83      0.82      0.83      3223
```