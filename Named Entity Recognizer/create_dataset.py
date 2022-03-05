# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:33:42 2021

@author: Utente
"""

import xml.etree.ElementTree as ET
import re


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def creation_sentences_dataset():
    tree = ET.parse('../dataset/sop_dataset.xml')
    root = tree.getroot()

    dataset = open("tweets_data.txt", "w", encoding="utf-8")
    file_indexes_nsop = open("index_nsop.txt", "w", encoding="utf-8")
    
    indexes_no_sop = []
    i = 0
    for elem in root:
        if elem[1].text == "Y":
            sentence = elem[0].text.replace("#", " ")
            sentence = remove_emoji(sentence)
            dataset.write(sentence + "\n")
        else:
            indexes_no_sop.append(i)
        i += 1
    
    for element in indexes_no_sop:
        file_indexes_nsop.write(str(element) + "\n")


    file_indexes_nsop.close()
    dataset.close()
        

creation_sentences_dataset()