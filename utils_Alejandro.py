import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gensim
import re
import spacy 
import gensim.models.word2vec as w2v
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

'''
    These are the main functions to manipulate
    and work around word2vect. They use regular
    expressions, and a pretrained pipeline from
    spacy to perform a series of operations to
    the data before introducing it into the 
    word2vec model.
'''

def cleaning(doc):
    '''
        Cleaning routine that lemmatizaises
        data.
    '''
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)

def preproces_train_data_Word2Vect(df, save=True):
    '''
        Preprocessing of the training data for the 
        word2vect data.
    '''
    #Drop nans
    df.dropna(inplace=True)
    #Build training data for Word2vect:
    #First separate question marks and words using regular expressions
    pattern = r"(\w+|[?!.])"
    sentences = list(np.append(df['question1'].values,df['question2'].values))
    sentences = [' '.join(re.findall(pattern, sent)) for sent in sentences]
    brief_cleaning = (re.sub("[^\w'?]+", ' ', str(row)).lower() for row in sentences)
    
    #Import spacy 
    nlp = spacy.load('en_core_web_sm') 
    
    #Use the spacy pipeline separating words etc
    txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000,  n_process=-1)]
    df_clean = pd.DataFrame({'clean': txt})
    df_clean = df_clean.dropna().drop_duplicates()
    if save:
        df_clean.to_csv('clean_w2v_data.csv', index=False)
    return df_clean

def preproces_data_Word2Vect(df):
    '''
        Cleaning the data that will be the
        input of the word2vect.
    '''
    string_columns = [col for col in df.columns if df[col].dtype == 'object']
    out = []
    for col in string_columns:
        sentences = list(df[col].values)
        #First separate question marks and words using regular expressions
        pattern = r"(\w+|[?!.])"
        sentences = list(np.append(df['question1'].values,df['question2'].values))
        sentences = [' '.join(re.findall(pattern, sent)) for sent in sentences]
        brief_cleaning = (re.sub("[^\w'?]+", ' ', str(row)).lower() for row in sentences)
        
        #Import spacy 
        nlp = spacy.load('en_core_web_sm') 
        
        #Use the spacy pipeline separating words etc
        txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000,  n_process=-1)]
        out.append(txt)
    return np.array(out)

def train_Word2Vect(df_clean=pd.read_csv('clean_w2v_data.csv'), num_features = 300, num_epochs = 20,
                    min_word_count = 0, num_workers = multiprocessing.cpu_count(), context_size = 5, 
                    downsampling = 1e-3, seed = 1, sg = 0, save=True):
    '''
        Function for training the word2vect from
        already cleaned data.
    '''
    
    word2vec = w2v.Word2Vec(
    sg=sg,
    seed=seed,
    workers=num_workers,
    vector_size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
    )
    
    word2vec.build_vocab(sent, progress_per=10000)
    word2vec.train(sent, total_examples=word2vec.corpus_count, epochs=30, report_delay=1)

    if save:
        word2vec.save("word2vec.model")
    return word2vec.wv

def sentence_to_wordlist(raw):
    '''
        Routine to separate sentence into
        wordlist.
    '''
    clean = re.sub("^a-zA-Z", " ", raw)
    clean = clean.lower()
    words = clean.split()
    return words
    
def doc_to_vec(sentence, word2vec):
    '''
        Routine to convert sentence into vector
        given a word2vect keyevector.
    '''
    word_list = sentence_to_wordlist(sentence)
    word_vectors = []
    for w in word_list:
        if w in word2vec.key_to_index.keys():
            word_vectors.append(word2vec[w])
    return np.mean(word_vectors, axis=0)

def get_Word2Vect(df, word2vec=Word2Vec.load("word2vec.model").wv, phrase2vec=doc_to_vec):
    '''
        Function that receives data to be cleaned and
        then converts it to vectors.
    '''
    data = preproces_data_Word2Vect(df, save=False)
    out = []
    if len(data.shape) >1:
        for val in data[:,0]:
            out.append(doc_to_vec(val,word2vec))
        array = np.array(out)
        for i in range(1, data.shape[1]):
            out = []
            for val in data[:,i]:
                out.append(doc_to_vec(val,word2vec))
            out = np.array(out)
            array = np.stack((array,out))
    else:
        for val in data:
            out.append(doc_to_vec(val,word2vec))
        array = np.array(out)
    return array

def get_Word2Vect_from_clean(df, word2vec=Word2Vec.load("word2vec.model").wv, phrase2vec=doc_to_vec):
    '''
        Function that receives celan data and
        then converts it to vectors.

        A suggeste input to word2vec is:
        word2vec = KeyedVectors.load("pretrained.model")

        "pretrained.model" is a keyevector from the 
        pretrained model: glove-wiki-gigaword-300
        you can also download it as such:
        
        gensim.downloader.load('glove-wiki-gigaword-300')
        
        I recommend saving with .save('file_path')
        as it will be faster for future use:
        
        pre.save("pretrained.model")
    '''
    data = df.values
    out = []
    if len(data.shape) >1:
        for val in data[:,0]:
            out.append(doc_to_vec(val,word2vec))
        array = np.array(out)
        for i in range(1, data.shape[1]):
            out = []
            for val in data[:,i]:
                out.append(doc_to_vec(val,word2vec))
            out = np.array(out)
            array = np.stack((array,out))
    else:
        for val in data:
            out.append(doc_to_vec(val,word2vec))
        array = np.array(out)
    return array