# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:42:25 2021

@author: Utente
"""

import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
from utils import *


def compute_sentiments(model, tweets_tuples):
    noOfTweet = len(tweets_tuples)
    positive = 0
    negative = 0
    neutral = 0
    polarity = 0
    tweets_predictions = []
    for tweet_tuple in tweets_tuples: 
        
        if model == "BLOB":       
            #metodo BLOB
            analysis = TextBlob(tweet_tuple[0])
            polarity = analysis.sentiment.polarity
            
            if polarity < 0:
                tweets_predictions.append((tweet_tuple[0], -1))
                negative += 1
            elif polarity == 0:
                tweets_predictions.append((tweet_tuple[0], 0))
                neutral += 1
            else:
                tweets_predictions.append((tweet_tuple[0], 1))
                positive += 1
            
        elif model == "VADER":
            #metodo Vader
            score = SentimentIntensityAnalyzer().polarity_scores(tweet_tuple[0])
            compound = score['compound']
            
            if compound >= 0:
                tweets_predictions.append((tweet_tuple[0], 1))
                positive += 1
            elif compound == 0:
                tweets_predictions.append((tweet_tuple[0], 0))
                neutral += 1 
            else:
                tweets_predictions.append((tweet_tuple[0], -1))
                negative += 1
        else:
            #metodo Vader 2
            score = SentimentIntensityAnalyzer().polarity_scores(tweet_tuple[0])
            neg = score['neg']
            pos = score['pos']
     
            if neg > pos:
                tweets_predictions.append((tweet_tuple[0], -1))
                negative += 1
            elif pos > neg:
                tweets_predictions.append((tweet_tuple[0], 1))
                positive += 1
     
            elif pos == neg:
                tweets_predictions.append((tweet_tuple[0], 0))
                neutral += 1
        
            
    positive = percentage(positive, noOfTweet)
    negative = percentage(negative, noOfTweet)
    neutral = percentage(neutral, noOfTweet)
    polarity = percentage(polarity, noOfTweet)
    positive = format(positive, '.1f')
    negative = format(negative, '.1f')
    neutral = format(neutral, '.1f')
    
    print("total number: ", noOfTweet)
    print("positive tweet: ", positive + "%")
    print("negative tweet: ", negative + "%")
    print("neutral tweet: ", neutral + "%")

    return tweets_predictions
    
    
#il dataset non è stato ripulito rimuovendo emoticons, @ e hastag poichè tali elementi possono influenzare il sentiment di un tweet.
#emoji come <3, :) hanno sentiment positivo, :( ha sentimento negativo, parole dopo # hanno un sentimento neutro
def main():
    MODEL = "VADER2"
    
    with open('sentiments_gold.tsv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = '\t')
        
        tweets_gold = []
        for entity in csv_reader:
           tweets_gold.append((entity[0], int(entity[1])))
    
    """            
    with open('tweets_london.tsv', encoding = "utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = '\t')
        
        tweets_gold = []
        for entity in csv_reader:
           tweets_gold.append((entity[0], None))"""
           
    tweets_pred = compute_sentiments(MODEL, tweets_gold)
    
    save_output_tsv(MODEL, tweets_pred)
    accuracy = accuracy_score(tweets_pred, tweets_gold)
    
    precision_positive = precision_score(1, tweets_pred, tweets_gold)
    precision_negative = precision_score(-1, tweets_pred, tweets_gold)
    
    recall_positive = recall_score(1, tweets_pred, tweets_gold)
    recall_negative = recall_score(-1, tweets_pred, tweets_gold)
    
    f1_positive = f1_score(precision_positive, recall_positive)
    f1_negative = f1_score(precision_negative, recall_negative)
      
    print("Accuracy: ", format(accuracy, '.5f'))
    
    print("\nANALYSIS SENTIMENT POSITIVE:")
    print("Precision positive: ", format(precision_positive, '.5f'))
    print("Recall positive: ", format(recall_positive, '.5f'))
    print("F1 score positive: ", format(f1_positive, '.5f'))
    
    print("\nANALYSIS SENTIMENT NEGATIVE:")
    print("Precision negative: ", format(precision_negative, '.5f'))
    print("Recall negative: ", format(recall_negative, '.5f'))
    print("F1 score negative: ", format(f1_negative, '.5f'))

    
        
if __name__ == "__main__":
    main()