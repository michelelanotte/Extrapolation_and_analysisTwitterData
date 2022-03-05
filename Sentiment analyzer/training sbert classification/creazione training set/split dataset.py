# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:55:15 2022

@author: Utente
"""
import pandas as pd

# Importing the dataset
DATASET_COLUMNS=['text', 'target']
DATASET_ENCODING = "ISO-8859-1"
data = pd.read_csv('dataset.tsv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS, sep="\t")

data_pos = data[data['target'] == 1] #800k tweet positivi
data_neg = data[data['target'] == 0] #800k tweet negativi

data_pos = data_pos.iloc[:int(6250)] 
data_neg = data_neg.iloc[:int(6250)]

dataset = pd.concat([data_pos, data_neg]).sample(frac=1)

dataset.to_csv('training_set.csv', index=False) 