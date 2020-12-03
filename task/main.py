import os,string
from ReadingAndPreprocessingData import ReadTsv, preprocess
from BalancingData import balanceData
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd
import numpy as np
import fasttext
from sklearn.model_selection import train_test_split
import sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import spacy

# This function simply prints a one D list of 
# predictions with ids of test data in respective files
def Output_in_file(filePath, y_pred, Test_data):
    out_f1 = open(filePath, 'w')
    out_f1.write('id, hateful\n')
    for i in range(len(Test_data)):
        out_f1.write(str(Test_data['id'][i]) + ', ' + str(y_pred[i]) + '\n')


def Test(f, Train_data, Test_data, RequireAnnData, name):
    # The following lines were used for checking 
    # the F1 scores, accuracy, etc. of the model
    from sklearn import metrics
    print("metrics for " + name)
    print(metrics.classification_report(Test_data['hateful'], f(*[Train_data, Test_data, RequireAnnData])))


# model 1
def RandomForestWithTfidf(Train_data, Test_data, RequireAnnData):
    # A vectorizer with specifications as mentioned in assignment
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5)

    # fitting on text of training data 
    vectorizer.fit(Train_data['text'])

    # getting our tfidf vector for text of training data and testing data
    X_train = vectorizer.transform(Train_data['text'])
    X_test = vectorizer.transform(Test_data['text'])
    X_Ann = vectorizer.transform(RequireAnnData['text'])

    # using Random forest classifier to 
    # predict the labels for testing data
    clf = RandomForestClassifier()
    clf.fit(X_train, Train_data['hateful'])

    y_pred = clf.predict(X_Ann)

    # printing the results in RF.csv
    Output_in_file('predictions/RF.csv', y_pred, RequireAnnData)

    return clf.predict(X_test)


# model 2
def SVCWithPretrainedEmbeddings(Train_data, Test_data, RequireAnnData):
    # Using pretrained embeddings of spacy 
    # to get the vectors for sentences
    nlp = spacy.load('en_core_web_md')

    # nlp(s).vector help get the vector for 
    # sentence which is mean of the vectors 
    # of the words in s
    X_train = np.array([nlp(s).vector for s in Train_data['text']])
    X_test = np.array([nlp(s).vector for s in Test_data['text']])
    X_Ann = np.array([nlp(s).vector for s in RequireAnnData['text']])

    y = Train_data['hateful']

    # a SVM classifier with default kernel as RBF
    clf = svm.SVC()

    # Training the SVM classifier and predicting the labels
    clf.fit(X_train, y)
    y_pred = clf.predict(X_Ann)
    
    # printing the results in SVM.csv
    Output_in_file('predictions/SVM.csv', y_pred, RequireAnnData)

    return clf.predict(X_test)


# model 3
def FasttextClassifier(Train_data, Test_data, RequireAnnData):
    # opening a temporary file
    subsf_1 = open('data/subs_train.tsv', 'w')
    Test_content = [s[:-1] for s in Test_data['text']]
    Ann_content = [s[:-1] for s in RequireAnnData['text']]

    # printing the contents of train_data in the temporary file 
    # with __label__ attached with labels as the 
    # fasttext classifier indentifies the labels like this only
    print(len(Train_data))
    print(Train_data)
    for i in range(len(Train_data)):
        subsf_1.write(Train_data['text'][i] + '\t' + '__label__' + str(Train_data['hateful'][i]) + '\n')

    # training the model on data in the temporary file
    model = fasttext.train_supervised('data/subs_train.tsv')
    
    # predicting the labels
    y_pred = model.predict(Ann_content)
    
    # printing the results in FT.csv
    Output_in_file('predictions/FT.csv', [int(pred[0][-1]) for pred in y_pred[0]], RequireAnnData)
    subsf_1.close()
    os.remove('data/subs_train.tsv')
    
    return [int(pred[0][-1]) for pred in model.predict(Test_content)[0]]


# model 4
def SGDClassifierWithPipeline(Train_data, Test_data, RequireAnnData):
    # resampling the hateful tweets to balance the data
    # this increases the efficiency of the model
    Train_data = balanceData(Train_data)

    # Using a SDG classifier as a pipeline 
    # with following hyperparameters and predicting the labels    
    pipeline_sgd = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf',  TfidfTransformer()),
        ('nb', SGDClassifier(loss="hinge", penalty="l2", max_iter=200)),])

    pipeline_sgd.fit(Train_data['text'], Train_data['hateful'])
    y_pred = pipeline_sgd.predict(RequireAnnData['text'])
   
    Output_in_file('predictions/SGD.csv', y_pred, RequireAnnData)

    return pipeline_sgd.predict(Test_data['text'])


if __name__ == "__main__":
    # This changes the directory to the 19EC10058 folder here 
    # which is important as the paths given in the functions is relative
    # to this folder only
    # on windows use '\\' instead of '/' in following line 
    os.chdir('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
    if(not os.path.exists('predictions')):
        os.mkdir('predictions')

    # pd.read_csv was not reading well 
    # so this custom function is used to 
    # read the training and testing files.
    Train_data = ReadTsv('data/train.tsv')
    Test_data = ReadTsv('data/test.tsv')

    # The custom preprocess function
    # helps 
    preprocess(Train_data)
    preprocess(Test_data)

    ############################################################################
    # following line is used for testing
    div = int(0.8*len(Train_data))
    train = Train_data[:div]
    test = Train_data[div:]
    ############################################################################

    # model 1
    Test(RandomForestWithTfidf, train, test, Test_data, 'RF')

    # model 2
    Test(SVCWithPretrainedEmbeddings, train, test, Test_data, 'SVM')

    # model 3
    Test(FasttextClassifier, train, test, Test_data, 'FT')

    #model 4
    Test(SGDClassifierWithPipeline, train, test, Test_data, 'SGD')