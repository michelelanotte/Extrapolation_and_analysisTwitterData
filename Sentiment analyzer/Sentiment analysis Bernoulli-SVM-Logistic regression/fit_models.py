# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:30:54 2021

@author: Utente
"""

# utilities
import pickle
import pandas as pd
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import *
from nltk.tokenize import word_tokenize


"""creazione dataset con solo testo e target"""
"""
# Importing the dataset
DATASET_COLUMNS=['id', 'text','target']
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv('dataset.tsv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS, sep="\t")

df['target'] = df['target'].replace([0], -1)
df.to_csv("dataset.tsv", sep="\t")"""



# Importing the dataset
DATASET_COLUMNS=['id', 'text', 'target']
DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv('dataset.tsv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS, sep="\t")

"""data_pos = data[data['target'] == 1] #800k tweet positivi
data_neg = data[data['target'] == -1] #800k tweet negativi

data_pos = data_pos.iloc[:int(500000)] 
data_neg = data_neg.iloc[:int(500000)]


dataset = pd.concat([data_pos, data_neg])"""

"""lower, rimozione di stopword, rimozione, punteggiatura, rimozione caratteri ripetuti, rimozione URLs, 
rimozione numeri, tokenizzazione, stemming e lemmatizzazione"""
dataset['text'] = preprocessing(dataset['text'])

    

#TRAINING SET
X_train = dataset.text.apply(lambda x: " ".join(x))
y_train = dataset.target

#vettorizzazione tf-idf. 148653 è la dimensione del vocabolario del training_set
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=148653)

#i tweet sono vettorizzati in formato tf-idf
X_train_matrix = vectoriser.fit_transform(X_train)


filename = "models/TfIdfvectorizer_model.sav"
pickle.dump(vectoriser, open(filename, "wb"))
#print('No. of feature_words: ', len(vectoriser.get_feature_names()))

"""
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
X_test_processed = X_train.tolist()
X_train_matrix  = sbert_model.encode(X_test_processed)"""


#---------modello di BernoulliNB-----------
BNBmodel = BernoulliNB()
BNBmodel.fit(X_train_matrix, y_train)
filename = "models/BNB_model.sav"
pickle.dump(BNBmodel, open(filename, "wb"))


#---------Support Vector Machine------------
SVCmodel = LinearSVC()
SVCmodel.fit(X_train_matrix, y_train)
filename = "models/SVM_model.sav"
pickle.dump(SVCmodel, open(filename, "wb"))


#---------Logistic regression------------
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train_matrix, y_train)
filename = "models/LR_model.sav"
pickle.dump(LRmodel, open(filename, "wb"))


#---------Multinomial NB------------
MNBmodel=MultinomialNB()
MNBmodel.fit(X_train_matrix, y_train)
filename = "models/MNB_model.sav"
pickle.dump(MNBmodel, open(filename, "wb"))