import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

data_dir = "./dataset/BGL"
log_name = "BGL_full.log_structured_0.8_rad_1.0"
dataset_file = os.path.join(data_dir,log_name+'.npz')
dataset = np.load(dataset_file, allow_pickle=True)

x_train = dataset["x_train"][()]
y_train = dataset["y_train"]
x_test = dataset["x_test"][()]
y_test = dataset["y_test"]

train_anomaly = 100 * sum(y_train) / len(y_train)
test_anomaly = 100 * sum(y_test) / len(y_test)

print("# train sessions: {} ({:.2f}%)".format(len(x_train), train_anomaly))
print("# test sessions: {} ({:.2f}%)".format(len(x_test), test_anomaly))

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

def getVectors(vectorizer, x_data, method):
    if method == "content":
        field = "Content"
    elif method == "template":
        field = "EventTemplate"
    X = []
    for key, evtList in x_data.items():

        aSeq = []
        for evt in evtList:
            aSeq.append(evt[field])
        aSeqVecs = vectorizer.transform(aSeq).toarray()

        aSeqVec = np.mean(aSeqVecs, axis=0)
        X.append(aSeqVec)
    X= np.array(X)

    return X

def Text_TFIDF_Generator(x_train, x_test, texttype):
    corp = getCorpus(x_train, texttype)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corp)
    batch_size = 1000

    len_train = len(x_train)
    n_bat = len_train/batch_size
    new_list = []
    for i in range(0,int(n_bat)+1):
        split_start = i*batch_size
        split_end = (i+1)*batch_size
        new_list.append({k: x_train[k] for k in list(x_train.keys())[split_start:split_end]})

    for i, dic in enumerate(new_list):
        print("Training set - Current: ",i*batch_size)
        if 'x_train_tmp' in vars():
            x_dic = getVectors(vectorizer, dic, texttype)
            x_train_tmp = np.concatenate((x_train_tmp,x_dic),axis=0)
        else:
            x_train_tmp = getVectors(vectorizer, dic, texttype)
    x_train = x_train_tmp

    # x_train = getVectors(vectorizer, self.x_train, texttype)
    len_test = len(x_test)
    n_bat = len_test/batch_size
    new_list = []
    for i in range(0,int(n_bat)+1):
        split_start = i*batch_size
        split_end = (i+1)*batch_size
        new_list.append({k: x_test[k] for k in list(x_test.keys())[split_start:split_end]})

    for i, dic in enumerate(new_list):
        print("Test set - Current: ",i*batch_size)
        if 'x_test_tmp' in vars():
            x_dic = getVectors(vectorizer, dic, texttype)
            x_test_tmp = np.concatenate((x_test_tmp,x_dic),axis=0)
        else:
            x_test_tmp = getVectors(vectorizer, dic, texttype)
    x_test = x_test_tmp


    return x_train, x_test

ttype = 'template'
x_train, x_test = Text_TFIDF_Generator(x_train, x_test, ttype)

save_path = os.path.join(data_dir,'TEXT_TFIDF_' + ttype + '_' + log_name + '.npz')

np.savez(save_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)