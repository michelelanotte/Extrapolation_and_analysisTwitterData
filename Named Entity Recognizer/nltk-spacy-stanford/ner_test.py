# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:07:41 2021

@author: Utente
"""

import csv
import spacy_ner
import nltk_ner
import stanford_ner
import utils
import spacy

def print_results(entities_dict, entities_gold_dict):
    precision = utils.precision(entities_dict, entities_gold_dict)
    recall = utils.recall(entities_dict, entities_gold_dict)
      
    print("Precision: ", precision)
    print("Recall: ", recall)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    print("F1 score: ", f1)


def main(): 
    nlp = spacy.load("en_core_web_sm")
    
    with open('ne_gold.tsv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = '\t')
        
        #indici dei tweet no-sop
        indexes_nop = utils.get_indexes_nsop_tweet()
        entities_gold_dict = {}
        tweets = []
        i = 0
        j = 0 #index for tweet sop
        for entity in csv_reader:
            if i not in indexes_nop:
                tweet = entity[0]
                tweet.replace("'s", '') #rimozione genitivo sassone
                
                #rimozione stopword
                tweet = utils.cleaningForNE(nlp, tweet)
                
                tweets.append(tweet)
                ne_list = entity[1:]
                
                ne_new_list = []
                #rimozione di elementi nulli nella lista di name entity
                for elem in ne_list:
                    
                    if elem != "":
                        ne_new_list.append(elem)
                
                entities_gold_dict[j] = ne_new_list
                j += 1

            i += 1
           
    """print("---------------SPACY---------------")
    entities_dict = spacy_ner.ner_spacy(tweets)
    print_results(entities_dict, entities_gold_dict)"""
    print("---------------NLTK---------------")
    entities_dict = nltk_ner.ner_nltk(tweets)
    print_results(entities_dict, entities_gold_dict)
    """print("---------------STANFORD---------------")
    entities_dict = stanford_ner.ner_stanford(tweets)
    print_results(entities_dict, entities_gold_dict)"""
    
    
if __name__ == "__main__":
    main()