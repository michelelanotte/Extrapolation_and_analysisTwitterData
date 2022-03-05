# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:17:04 2021

@author: Utente
"""

import spacy
    

def ner_spacy(tweets):
    filters = ["PERSON", "TIME", "DATE", "ORDINAL", "MONEY", "WORK_OF_ART", "CARDINAL", "PRODUCT", "PERCENT", "EVENT"]
    nlp = spacy.load("en_core_web_sm")
    

    #dizionario dove la key è un indice che identifica il tweet e i valori sono liste di entità
    entities_dict = {}
    i = 0
    for tweet in tweets:        
        new_doc = nlp(tweet)
        
        entities_dict[i] = []
        for ent in new_doc.ents:
            text = ent.text
            if "https" not in text:
                if ent.label_ not in filters:
                    entities_dict[i].append(ent.text)
        i += 1
    return entities_dict