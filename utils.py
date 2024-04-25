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

##################### begin utils from utils_Alba #########################

# all lower case sentence
def lowercase_sentence(sentence):
    """
    Args:
    sentence (str): The input sentence to be put in lower case.

    Returns:
    list: A list of tokens extracted from the input sentence.
    """
    new_sentence = sentence.lower()
    return new_sentence


# remove punctuation from a sentence
def remove_punctuation(sentence):
    """
    Args:
    sentence (str): The input sentence to remove punctuation.

    Returns:
    list: The sentence without punctuation symbols.
    """    
    new_sentence = re.sub(r'[^\w\s]', '', sentence) # matches non words and non spaces (includes '?') 
    return new_sentence


# remove accents
def remove_accents(sentence):
    '''
    Args:
      sentence (str): The input sentence to remove accent
    Return:
      str : The sentence without accents
    '''
    new_sentence = unidecode.unidecode(sentence) 
    return new_sentence


# remove non-alpha characters and non-alphanumeric characters (that is, special characters: punctuation marks, spaces, accents)
def remove_special_characters(sentence, numeric = False):
    """
    Args:
    sentence (str): The input sentence to remove non-alphanumeric characters.
    numeric (bool): if true, numbers are also removed

    Returns:
    str: The sentence without non-alphanumric characters (includes punctuation symbols and spaces).
    """
    if numeric:
        new_sentence = re.sub(r'[^a-zA-Z]', ' ', sentence) # matches non-alpha characters 
    else:
        new_sentence = re.sub(r'[^a-zA-Z0-9]', ' ', sentence) # matches non-alphanumeric characters
    return new_sentence


# remove stop words
def remove_stopwords(sentence):
    """
    Args:
    sentence (str): The input sentence from which stop words will be removed.

    Returns:
    str: The input sentence with stop words removed.
    """
    #stop_words = set(stopwords.words('english')) # predefined stop words in English
    stop_words = set(['the', 'and', 'to', 'in', 'of', 'that', 'is', 'it', 'for',
    'on', 'this', 'you', 'be', 'are', 'or', 'from', 'at', 'by', 'we',
    'an', 'not', 'have', 'has', 'but', 'as', 'if', 'so', 'they', 'their',
    'was', 'were','some', 'there', 'these', 'those', 'than', 'then', 'been', 'also',
    'much', 'many', 'other']) # custom defined set
    
    words = nltk.word_tokenize(sentence)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    new_sentence = ' '.join(filtered_words)
    return new_sentence


# Normalize spaces - Replace all consecutive whitespace characters in the text string with a single space.
def normalize_spaces(sentence):
    '''
    Args:
      sentence (str): The input sentence to normalize
    Returns:
      str: The final sentence normalized 
    '''
    new_sentence = re.sub(r'\s+', ' ', sentence)
    return new_sentence

# Number of words in a sentence
def number_words(sentence):
    '''
    Args:
      sentence (str): The input sentence to count the number of words
      
    Returns:
      int : The number of words in the given text
    '''
    return len(word_tokenize(sentence))


# Number of common words between two sentences
def number_common_words(s1, s2):
    '''
    Args:
      s1 (str): First sentence
      s2 (str): Second sentence
    
    Return:
      int: The number of common words that the two sentences have in common
    '''
    # Tokenize
    tokens1 = set(word_tokenize(s1))
    tokens2 = set(word_tokenize(s2))
    
    common = tokens1 & tokens2 # list of common tokens
    return len(common)


# Number of common words in the same position
def number_common_words_2(s1, s2):
    """
    Args:
      s1 (str): The first input sentence.
      s2 (str): The second input sentence.

    Returns:
      int: The number of common words at the same position in both sentences.
    """
    # Tokenize
    tokens1 = word_tokenize(s1)
    tokens2 = word_tokenize(s2)

    min_length = min(len(tokens1), len(tokens2))

    # Common words at the same position
    common_count = 0
    for i in range(min_length):
        if tokens1[i].lower() == tokens2[i].lower():
            common_count += 1

    return common_count


# If the first word of two sentences is equal
def first_word_equal(s1, s2):
    """
    Args:
      s1 (str): First sentence
      s2 (str): Second sentence
    Returns:
      A binary value indicating whether the firsts words of the two questions are equal.
    """
    # Tokenize
    tokens1 = word_tokenize(s1)
    tokens2 = word_tokenize(s2)
    
    if tokens1[0].lower() == tokens2[0].lower():
            return 1
    
    return 0


# If the last word of two sentences is equal
def last_word_equal(s1, s2):
    """
    Args:
      s1 (str): First sentence
      s2 (str): Second sentence
    Returns:
      A binary value indicating whether the lasts words of the two questions are equal.
    """
    # Tokenize
    tokens1 = word_tokenize(s1)
    tokens2 = word_tokenize(s2) # with word_tokenize, counts '.' as different token
    
    if tokens1[-1].lower() == tokens2[-1].lower():
            return 1
    
    return 0

# stemming (using both methods) -> remove prefixes and suffixes, may return non existing word 
def stem(sentence, type_porter = True):
    '''
    Args:
      sentence (str): The input sentence for stemming
      type_porter (bool): if True we use the Porter method, if false, the Lancaster method
    Returns:
      str: The final sentence stemmed
    '''
    token_words = word_tokenize(sentence)
    sentence_stemmed = []
    if type_porter:
        for word in token_words:
            sentence_stemmed.append(porter.stem(word))
            sentence_stemmed.append(" ")
    else:
        for word in token_words:
            sentence_stemmed.append(lancaster.stem(word))
            sentence_stemmed.append(" ")
    return "".join(sentence_stemmed)


# lemmantization (using wordnet_lemmatizer.lemmatize(w)) -> remove endings to return base word (it is a valid word)
def lemma(sentence):
    '''
    Args:
      sentence (str): The input sentence for lemmantization
      str: The final sentence lemmantized
    '''
    token_words = word_tokenize(sentence)
    sentence_lemma = []
    for word in token_words:
        sentence_lemma.append(wordnet_lemmatizer.lemmatize(word)) # focus on verbs
        sentence_lemma.append(" ")
    return "".join(sentence_lemma)

def preprocess_data(question_list):
    """
    Args:
      question_list (str list): list of string questions
    Returns:
      The list after preprocessing (we apply the preprocessing functions).
    """
    q_lower = [lowercase_sentence(question) for question in question_list] # lovercase
    q_sc = [remove_special_characters(question) for question in q_lower] # remove special characters (all)
    q_sw = [remove_stopwords(question) for question in q_sc] # remove stop words
    q_preprocessed = [normalize_spaces(question) for question in q_sw] # normalize spaces
    return q_preprocessed

def build_numeric_features(q1_list, q2_list):
    """
    Args:
      q1_list (str list): list of string questions
      q2_list (str list): list of string questions
    Returns:
      A data frame containing the text features applied to both lists.
    """
    # number of words
    q1_f1 = [number_words(question) for question in q1_list]
    q2_f1 = [number_words(question) for question in q2_list]
    
    # number of common words
    q1q2_f2 = [number_common_words(question1, question2) for question1, question2 in zip(q1_list, q2_list)]
    
    # number of common words in the same position
    q1q2_f3 = [number_common_words_2(question1, question2) for question1, question2 in zip(q1_list, q2_list)]
    
    # first word equal
    q1q2_f4 = [first_word_equal(question1, question2) for question1, question2 in zip(q1_list, q2_list)]
    
    # last word equal
    q1q2_f5 = [last_word_equal(question1, question2) for question1, question2 in zip(q1_list, q2_list)]
    
    # build dataframe with features
    df_features = pd.DataFrame({'num_words_1': q1_f1, 'num_words_2': q2_f1, 'num_common_words': q1q2_f2,
                               'num_common_words_2': q1q2_f3, 'first_word': q1q2_f4, 'last_word': q1q2_f5}, 
                               columns=['num_words_1', 'num_words_2', 'num_common_words', 'num_common_words_2', 
                                        'first_word', 'last_word'])
    return df_features


##################### end utils from utils_Alba #########################