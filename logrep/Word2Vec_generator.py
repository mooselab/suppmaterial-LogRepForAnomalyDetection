import json
import gensim
import gensim.downloader as api
from nltk.tokenize import RegexpTokenizer
import re
import numpy as np
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


tokenizer = RegexpTokenizer(r'\w+')

path = api.load("word2vec-google-news-300", return_path=True)
wv = api.load('word2vec-google-news-300')

# load splitted dataset
splitted_dataset = np.load('./dataset/BGL/BGL_full.log_structured_0.8_rad_6.0.npz', allow_pickle=True)

x_train = splitted_dataset["x_train"][()]
y_train = splitted_dataset["y_train"]
x_test = splitted_dataset["x_test"][()]
y_test = splitted_dataset["y_test"]

def getCorpus(x_data, method):

    if method == "content":
        field = "Content"
    elif method == "template":
        field = "EventTemplate"

    corpus = []
    for key, evtList in x_data.items():
        text = ""
        for evt in evtList:
            text = text + evt[field]
        corpus.append(text)

    return corpus

corp = getCorpus(x_train, "template")
vectorizer = TfidfVectorizer()
vectorizer.fit(corp)

def word2vec_generator(x_data, ft, s_parse, tfidf_vectorizer = None):
    x_data_vec = []
    if not s_parse:
        input_type = "Content"
    else:
        input_type = "EventTemplate"
    
    if tfidf_vectorizer != None:
        tfidf_voc = tfidf_vectorizer.vocabulary_
    
    blks = []
    for blk,seq in tqdm(x_data.items()):
        sens_list = []
        for event in seq:
            s = event[input_type]
            sentence = tokenizer.tokenize(s)
            vectors = [wv[w] if w in wv else wv['UNK'] for w in sentence ]
            if tfidf_vectorizer != None:
                tfidf_val = tfidf_vectorizer.transform([s]).toarray()
#                 tfidf_vec = [1 for w in sentence]
                tfidf_vec = [tfidf_val[0][tfidf_voc[w.lower()]] if w.lower() in tfidf_voc else 1 for w in sentence]
                vec_sen = np.dot(tfidf_vec, vectors)
                if vec_sen.any()==False:
                    vec_sen = np.zeros(300)
            else:
                if len(vectors)==0:
                    vec_sen = np.zeros(300)
                else:
                    vec_sen = np.mean(vectors, axis=0) #np.mean
            sens_list.append(vec_sen)
        sens_list = np.array(sens_list,dtype=object)
#         print(sens_list.shape)
        blks.append(sens_list)
    x_data_vec= np.array(blks,dtype=object)
    
    print(x_data_vec.shape)
    return x_data_vec

x_train_feature = word2vec_generator(x_train, wv, True, vectorizer)
x_test_feature = word2vec_generator(x_test, wv, True, vectorizer)

np.savez('./dataset/BGL/BGL_word2vec_template_TFIDF_300d.npz',x_train = x_train_feature, y_train = y_train, x_test=x_test_feature, y_test=y_test)