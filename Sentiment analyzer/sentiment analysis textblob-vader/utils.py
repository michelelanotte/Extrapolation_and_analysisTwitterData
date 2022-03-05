# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:05:25 2021

@author: Utente
"""

import re
import csv

def percentage(part, whole):
    return 100 * float(part)/float(whole)


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


def cleaning_text(tweet):

    #Removing RT and @
    tweet = re.sub("(@[A-Za-z0–9]+)|RT @\w+: ",' ',tweet)
    
    tweet = tweet.replace("#", " ")
    tweet = remove_emoji(tweet)
    tweet = tweet.lower()
    return tweet


"""tweets_pred e tweets_gold sono liste di tuple dove il primo elemento di una tupla è il tweet i-esimo e il secondo elemento
coincide con la predizione del sentiment(per tweets_pred) o con il sentiment annotato(per tweet_gold)"""
def accuracy_score(tweets_pred, tweets_gold):
    num_tweets = len(tweets_pred)
    tp = 0
    for i, tweet_entity in enumerate(tweets_pred):
        if tweet_entity[1] == tweets_gold[i][1]:
            
            tp += 1
        
    return tp / num_tweets


def compute_num_tweets(polarity, tweets_pred):
    counter = 0
    for elem in tweets_pred:
        if elem[1] == polarity:
            counter += 1
    return counter
    

def precision_score(polarity, tweets_pred, tweets_gold):
    num_tweets = compute_num_tweets(polarity, tweets_pred)
    tp = 0
    for i, tweet_entity in enumerate(tweets_pred):
        if tweets_gold[i][1] == polarity and tweet_entity[1] == polarity:
                tp += 1
    return tp/num_tweets

    
def recall_score(polarity, tweets_pred, tweets_gold):
    num_tweets = compute_num_tweets(polarity, tweets_gold)
    tp = 0
    for i, tweet_entity in enumerate(tweets_pred):
        if tweets_gold[i][1] == polarity and tweet_entity[1] == polarity:
                tp += 1
    return tp/num_tweets

    
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def save_output_tsv(model, tweets_pred):
    with open("../predictions/london_prediction_" + model + ".tsv", mode = "w") as out_file:
        for prediction in tweets_pred:
            if prediction[1] == 1:
                sentiment = "POSITIVE"
            elif prediction[1] == -1:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
            tweet = cleaning_text(remove_emoji(prediction[0]))
            out_file.writelines(tweet + "\t" + sentiment + "\n")