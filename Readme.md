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
4. A simple SGD Classifier

For using, just clone the repo and then run ./setup.sh in the directory of the repo.<br>
This will install the required files.