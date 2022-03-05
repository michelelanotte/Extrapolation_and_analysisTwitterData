# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:27:31 2021

@author: Utente
"""

from sklearn.metrics import precision_score
import numpy as np


def get_indexes_nsop_tweet():
    indexes = []
    
    with open('index_nsop.txt') as file:
        lines = file.readlines()
    
    for elem in lines:
        indexes.append(int(elem))
    return indexes



def stopword_removal(nlp, doc):
    # Create list of word tokens after removing stopwords
    filtered_sentence =[] 
    
    token_list = []
    for token in doc:
        token_list.append(token.text)

    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word) 
    return " ".join(filtered_sentence) 


def count_en_gold(entities_gold_dict):
    counter = 0
    for en_key in entities_gold_dict:
        counter += len(entities_gold_dict[en_key])
    
    return counter
   

def precision(entities_dict, entities_gold_dict):
    num_tweets = len(entities_dict.keys())
    partial_precision = 0
    #tp_set = set()
    #fp_set = set()
    for tweet_entities in entities_dict:
        set_gold = set(entities_gold_dict[tweet_entities])
        set_predictions = set(entities_dict[tweet_entities])
        
        #tp_set.update(set_predictions.intersection(set_gold))
        #fp_set.update(set_predictions)
        tp = len(set_predictions.intersection(set_gold))
        
        denominator = len(set_predictions)
        
        #se il denomiantore è zero alla partial_precision non si aggiunge nulla poichè la precision è zero
        if denominator != 0:
            partial_precision += tp/denominator
    
    #print("tp:", tp_set)
    #print("fp:", fp_set)
        
    return partial_precision / num_tweets


def recall(entities_dict, entities_gold_dict):
    num_tweets = len(entities_dict.keys())
    partial_recall = 0
    denominator = 0 #tp+fn
    fn_set = set()
    for tweet_entities in entities_dict:
        set_gold = set(entities_gold_dict[tweet_entities])
        set_predictions = set(entities_dict[tweet_entities]) 
        
        tp = len(set_predictions.intersection(set_gold))
        fn_set.update(set_gold.difference(set_predictions))
        denominator = len(set_gold)  
        
        if denominator != 0:
            partial_recall += tp/denominator    
    return partial_recall / num_tweets



import re
import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)


def cleaning_URLs(text):
    pattern_re = "https?:\S+|http?:\S|[^A-Za-z0-9]+"
    return re.sub(pattern_re, ' ', text)


def cleaning_tags(text):
    pattern_re = "@\S+|RT @\S+"
    return re.sub(pattern_re, '', text)  


def cleaning_URLs_and_tagging(text):
    text = cleaning_URLs(text)
    return cleaning_tags(text)


def cleaning_numbers(text):
    return re.sub('[0-9]+', '', text)


def cleaningForNE(nlp, tweet):
    tweet = cleaning_tags(tweet)
    tweet = tweet.replace("’s", '')
    
    tweet_nlp = nlp(tweet)
    tweet = stopword_removal(nlp, tweet_nlp)
    
    tweet = tweet.replace("...", "")
    tweet = tweet.replace("â", "")
    return cleaning_numbers(tweet)