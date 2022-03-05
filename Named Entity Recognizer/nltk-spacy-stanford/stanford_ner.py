# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:08:05 2021

@author: Utente
"""

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import os

os.environ['JAVAHOME'] =  "C:/Program Files/Java/jdk-11.0.6/bin/java.exe"

st = StanfordNERTagger('stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   'stanford-ner/stanford-ner-4.2.0.jar',
					   encoding='utf-8')


def filtering(classified_text):
    filters = ["O", "PERSON", "ORGANIZATION"]
    ne_list = []
    for elem in classified_text:
        if elem[1] not in filters:
            ne_list.append(elem[0])
    return ne_list


def ner_stanford(tweets):
    entities_dict = {}
    i = 0
    for tweet in tweets: 
        print(i)
        tokenized_text = word_tokenize(tweet)
        classified_text = st.tag(tokenized_text)
        ne_list = filtering(classified_text)
        entities_dict[i] = ne_list
        i += 1
    return entities_dict