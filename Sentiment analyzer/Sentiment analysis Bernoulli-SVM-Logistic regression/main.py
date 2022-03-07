# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:46:58 2021

@author: Utente
"""

import pickle
from utils import *
import pandas as pd

DATASET_COLUMNS=['text', 'target']
DATASET_ENCODING = "ISO-8859-1"

#TEST SET
data_test = pd.read_csv('sentiments_gold.tsv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS, sep="\t")
X_test = data_test.text
y_test = data_test.target
output_file = "../predictions/prediction_"



"""#TEST SET LONDON
data_test = pd.read_csv('tweets_london.tsv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS, sep="\t")
X_test = data_test.text
y_test = None
output_file = "../predictions/london_prediction_"""


#PREPROCESSING TEST_SET
"""lower, rimozione di stopword, rimozione, punteggiatura, rimozione caratteri ripetuti, rimozione URLs, 
rimozione numeri, tokenizzazione, stemming e lemmatizzazione"""
X_test_processed = preprocessing(X_test)


#vettorizzazione
filename = "models/TfIdfvectorizer_model.sav"
vectoriser = pickle.load(open(filename, 'rb'))
X_test_vecs = vectoriser.transform(X_test_processed.apply(lambda x: " ".join(x)))

"""
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
X_test_processed = X_test_processed.apply(lambda x: " ".join(x)).tolist()
X_test_vecs  = sbert_model.encode(X_test_processed)"""

print("-------BERNOULLI--------")
filename = "models/BNB_model.sav"
BNBmodel = pickle.load(open(filename, 'rb'))
model_Evaluate(BNBmodel, X_test, X_test_vecs, output_file, y_test)

print("-------SVM--------")
filename = "models/SVM_model.sav"
SVCmodel = pickle.load(open(filename, 'rb'))
model_Evaluate(SVCmodel, X_test, X_test_vecs, output_file, y_test)

print("-------Logistic regression--------")
filename = "models/LR_model.sav"
LRmodel = pickle.load(open(filename, 'rb'))
model_Evaluate(LRmodel, X_test, X_test_vecs, output_file, y_test)

print("-------Multinomial NB--------")
filename = "models/MNB_model.sav"
MNBmodel = pickle.load(open(filename, 'rb'))
model_Evaluate(MNBmodel, X_test, X_test_vecs, output_file, y_test)