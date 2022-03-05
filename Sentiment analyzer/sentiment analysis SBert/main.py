# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:46:58 2021

@author: Utente
"""

import torch
from utils import *
import pandas as pd
from dataset import *
from tqdm import tqdm

DATASET_COLUMNS=['text', 'target']
DATASET_ENCODING = "ISO-8859-1"

#TEST SET
data_test = pd.read_csv('sentiments_gold.tsv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS, sep="\t")
tweets = data_test.text
labels = data_test.target
output_file = "../predictions/prediction_"


#PREPROCESSING TEST_SET
"""lower, rimozione caratteri ripetuti, rimozione URLs"""
processed_tweets = preprocessing(tweets)


model_sbert = torch.load('models/sbertweet200kno-freeze.pth', map_location=torch.device('cpu'))

device = torch.device("cpu")
val = Dataset(data_test)
val_dataloader = torch.utils.data.DataLoader(val)
labels_gold = []
predictions = []
for val_input, val_label in tqdm(val_dataloader):
    val_label = val_label.to(device)
    mask = val_input['attention_mask'].to(device)
    input_id = val_input['input_ids'].squeeze(1).to(device)

    output = model_sbert(input_id, mask)
    
    predictions.append(output.argmax(dim=1).item())
    labels_gold.append(val_label.item())


model_name = type(model_sbert).__name__
output_file = "prediction_"
model_Evaluate(model_name, tweets, output_file, predictions, labels_gold)