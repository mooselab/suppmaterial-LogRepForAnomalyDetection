import json
import gensim
import numpy as np
from tqdm.notebook import tqdm

splitted_dataset = np.load('./dataset/BGL/BGL_full.log_structured_0.8_rad_6.0.npz', allow_pickle=True)
x_train = splitted_dataset["x_train"][()]
y_train = splitted_dataset["y_train"]
x_test = splitted_dataset["x_test"][()]
y_test = splitted_dataset["y_test"]

from nltk.tokenize import RegexpTokenizer
import fasttext.util
import numpy as np
ft = fasttext.load_model('cc.en.300.bin')
fasttext.util.reduce_model(ft, 50)
tokenizer = RegexpTokenizer(r'\w+')


from sklearn.feature_extraction.text import TfidfVectorizer

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


def fasttext_generator(x_data, ft, s_parse, tfidf_vectorizer = None):
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
            vectors = [ft.get_word_vector(w) for w in sentence]
            if tfidf_vectorizer != None:
                tfidf_val = tfidf_vectorizer.transform([s]).toarray()
#                 tfidf_vec = [1 for w in sentence]
                tfidf_vec = [tfidf_val[0][tfidf_voc[w.lower()]] if w.lower() in tfidf_voc else 1 for w in sentence]
                vec_sen = np.dot(tfidf_vec, vectors)
                if vec_sen.any()==False:
                    vec_sen = np.zeros(ft.get_dimension())
            else:
                if len(vectors)==0:
                    vec_sen = np.zeros(ft.get_dimension())
                else:
                    vec_sen = np.mean(vectors, axis=0) #np.mean
            sens_list.append(vec_sen)
        sens_list = np.array(sens_list,dtype=object)
#         print('sl_shape',sens_list.shape)
        if len(sens_list.shape)==1:
            print(sens_list)
        blks.append(sens_list)
    x_data_vec= np.array(blks,dtype=object)
    
#     print(x_data_vec.shape)
    return x_data_vec

x_train_feature = fasttext_generator(x_train, ft, True, vectorizer)
# x_train_feature = fasttext_generator(x_train, ft, False)
x_test_feature = fasttext_generator(x_test, ft, True, vectorizer)
# x_test_feature = fasttext_generator(x_test, ft, False)

np.savez('./dataset/BGL/BGL_fasttext_template_tfidf_50d_0.8_6h.npz',x_train = x_train_feature, y_train = y_train, x_test=x_test_feature, y_test=y_test)