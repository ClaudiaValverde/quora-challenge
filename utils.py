import scipy
import sklearn
from sklearn import *
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings
    
    
def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted =  cast_list_as_strings(list(df["question1"]))
    q2_casted =  cast_list_as_strings(list(df["question2"]))
    
    ############### Begin exercise ###################
    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)    
    X_q1q2 = scipy.sparse.hstack((X_q1,X_q2))
    ############### End exercise ###################

    return X_q1q2
    
def get_mistakes(clf, X_q1q2, y):
    '''
    Make a function get_mistakes that given a model clf a dataframe df, the features X_q1q2 and the target labels yreturns
    - incorrect_indices: coordinates where the model made a mistake
    - predictions: predictions made by the model
    '''

    ############### Begin exercise ###################
    predictions = clf.predict(X_q1q2)
    incorrect_predictions = predictions != y 
    incorrect_indices,  = np.where(incorrect_predictions)
    
    ############### End exercise ###################
    
    if np.sum(incorrect_predictions)==0:
        print("no mistakes in this df")
    else:
        return incorrect_indices, predictions
    
# our custom functions

def evaluate_model(X, y_true, model, display=False):
    '''
    function that evaluates the trained model. It returns of ROC AUC score, precision, recall as well as accuracy and f1.
    
    Args: 
    - X: data to predict
    - y_true: labels of the data
    - model: trained model to evaluate
    - display: is a boolean, if True it will print all evaluation metrics and grafics just by calling the function. If false, it will only return metrics ditionary.
    
    Returns:
    - y_pred: predictions
    - Metrics dict:
        Accuracy
        F1
        Precision
        Recall
        ROC AUC score
        ##########should we do log-loss????????
    - classificacion report with precision, recall, acc and f1 for each class
    - plots a confusion matrix
    '''
    metrics = {}
    # predict
    y_pred = model.predict(X)

    # metrics
    accuracy = model.score(X, y_true)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)
    
    metrics = {'accuracy': accuracy, 'roc_auc':roc_auc, 'precision':precision, 'recall':recall, 'f1':f1}
    
    if display==True:
        print('METRICS:', '\n', '---'*30)
        print('Accuracy: ', metrics['accuracy'])
        print('F1: ',metrics['f1'])
        print('Precision: ', metrics['precision'])
        print('Recall: ', metrics['recall'])
        print('ROC AUC: ', metrics['roc_auc'])
        
    
        # printint classification report with acc, f1, precision and recall
        print('\n CLASSIFICATION REPORT:', '\n', '---'*30)
        print(sklearn.metrics.classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

        # ploting confusion matrix
        print('CONFUSION MATRIX:', '\n', '---'*30)
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    return metrics