# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:24:33 2021

@author: Utente
"""

import nltk
import spacy
import utils
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree


def ner_nltk(tweets):
    filters =  ["LOCATION", "GPE", "ORGANIZATION"]
    #dizionario dove la key è un indice che identifica il tweet e i valori sono liste di entità
    entities_dict = {}
    i = 0
    #nlp = spacy.load("en_core_web_sm")
    for tweet in tweets:
        entities_dict[i] = []
        chunked = ne_chunk(pos_tag(word_tokenize(tweet)))
        current_chunk = []
        for node in chunked:
            if type(node) is Tree:
                if node.label() in filters:
                    leaves = node.leaves()
                    named_entity = " ".join(x[0] for x in leaves)
                    if named_entity not in entities_dict[i]:
                        entities_dict[i].append(named_entity)
        
        i += 1
    return entities_dict