# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 10:43:17 2021

@author: Utente
"""

import xml.etree.ElementTree as ET
from geopy import Nominatim
import googlemaps
import geocoder
import pandas as pd
import re
import spacy
import string
import nltk
from nltk.tokenize import word_tokenize
import geopy.distance
#from nltk.corpus import stopwords

#stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

BING_KEY = "AjttQre1RsLFdGceLZYGGUWx0f3NY3ZyJiUU7tbTtWalvxVhSXhGH1kd1mMh0KzB"


def creation_sentences_dataset():
    dataset = open("dataset/tweets_data.txt", "w", encoding="utf-8")
    
    tree1 = ET.parse('dataset/sop_dataset.xml')
    root1 = tree1.getroot() 
    write_tweet(root1, dataset)

    #aggiunta tweet sop di londra
    tree2 = ET.parse('dataset/london.xml')
    root2 = tree2.getroot()
    write_tweet(root2, dataset)
    
    dataset.close()
    

def write_tweet(root, dataset):
    for elem in root:
        if elem[1].text == "Y":
            sentence = elem[0].text
            dataset.write(sentence + "\n")


def clean_text(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        sentence = tweet.replace("#", '').replace("’s", '')   #rimozione genitivo sassone
        sentence = re.sub("(@[A-Za-z0–9]+)|RT @\w+: ",' ', sentence).replace("@", "")
        sentence = re.sub(r'http\S+',' ', sentence)
        sentence = remove_emoji(sentence)
        cleaned_tweets.append(cleaning_stopwords(sentence))
    return cleaned_tweets


def getNE(tweet, filters):
    new_doc = nlp(tweet)
        
    entities = []
    for ent in new_doc.ents:
        text = ent.text
        if "https" not in text:
            if ent.label_ not in filters:
                entities.append(ent.text)
    return entities


def read_tsv(tsv):
    df = pd.read_csv("predictions/" + tsv, delimiter = "\t", encoding='cp1252', names = ["Tweet", "Sentiment"])
    return df


"""
This method returns first hastag in tweet.
"""
def getFirstHastag(tweet):
    res = ""
    if "#" in tweet:
        pattern_re = "#\w+"
        span = re.search(pattern_re, tweet).span()
        res = tweet[span[0]+1:span[1]]
    return res


def getHastags(tweet):
    res = []
    if "#" in tweet:
        pattern_re = "#\w+"
        res = re.findall(pattern_re, tweet)
    return res


"""
This method write triples <tweet, sentiment, coordinate> in tsv file specified in the arguments
"""
def dataFrameToTsv(dataset, tsv_file):
    df = pd.DataFrame(dataset, columns = ["Tweet", "Sentiment", "Coordinate"]) 
    df.to_csv(tsv_file, sep = "\t", index = False, line_terminator='\n')


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642" 
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def cleaning_stopwords(tweet):
    tweet_nlp = nlp(tweet)
    filtered_tweet = []
    
    token_list = []
    for token in tweet_nlp:
        token_list.append(token.text)
    for word in token_list:
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_tweet.append(word) 
            
    return " ".join(filtered_tweet) 


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


st = nltk.PorterStemmer()
def stemming_on_text(tokens):
    text = [st.stem(word) for word in tokens]
    return text

lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(tokens):
    text = [lm.lemmatize(word) for word in tokens]
    return text


def cleaning_tweet(tweet):
    #print(tweet)
    tweet = cleaning_URLs_and_tagging(tweet)
    tweet = cleaning_stopwords(tweet)
    tweet = cleaning_punctuations(tweet)
    tweet = cleaning_repeating_char(tweet)
    tweet = cleaning_numbers(tweet)
    
    #tokenization is used for stemming and lemmatization
    tokenized_tweets = word_tokenize(tweet)
    tweet = stemming_on_text(tokenized_tweets)
    
    #print(tweet)
    #print("*********************************************")
    return lemmatizer_on_text(tweet)


"""
With this method, tweets are cleared of stopwords, punctuation, URLs, repeated characters, numbers. 
Stemming and lemmatization are also applied to tweets. This method is used for sentiment analysis
Method return DataSeries of cleaned tweets
"""
def preprocessing(tweets):
    cleaned_tweets = [cleaning_tweet(tweet) for tweet in tweets]
    return pd.Series(cleaned_tweets)


"""
With this method, tweet are cleared of stopwords, numbers, tags. 
Stemming and lemmatization are also applied to tweets. This method is used for sentiment analysis
Method return DataSeries of cleaned tweets
"""
def cleaningForNE(tweet):
    tweet = cleaning_tags(tweet)
    tweet = tweet.replace("’s", '')
    tweet = cleaning_stopwords(tweet)
    tweet = tweet.replace("...", "")
    tweet = tweet.replace("â", "")
    tweet = tweet.replace("#", "")
    return cleaning_numbers(tweet)


def prefilteringCoordinates(places, coord_couples, coord_bbox):
    new_coord_list = []
    candidate_places = []
    for i, bbox1 in enumerate(coord_bbox):
        contain_places = False
        
        #check if bbox1 contain another bbox
        for j, bbox2 in enumerate(coord_bbox):
            if i != j:             
                #It's verified that the bbox2 has latitudes between the latitudes of bbox1
                if bbox1[0][0] <= bbox2[0][0] and bbox2[0][1] <= bbox1[0][1]:
                    #It's verified that the bbox2 has longitudes between the longitudes of bbox1
                    if bbox1[1][0] <= bbox2[1][0]  and bbox2[1][1] <= bbox1[1][1]:
                        contain_places = True
                        break
                    
        #If the i-th bbox does not contain places, then the i-th coordinates are added to the output list
        if not contain_places:
            new_coord_list.append(coord_couples[i])
            candidate_places.append(places[i])
    return candidate_places, new_coord_list
                    
    
    
def getCoordFromPlace(tweet, places, heuristic):
    coordinate = None
            
    if heuristic == 3:
        if places:
            place = " ".join(places)
            position = geocoder.bing(place, key=BING_KEY) 
            if position.json:
                lat_lon = (position.json['lat'], position.json['lng'])
                coordinate = lat_lon
    else:
        #form of elem in list <(lat_min,lat_max), (lon_min, lon_max)>.
        coord_bbox = []
        
        #form of elem in list <lat, lon>
        coord_couples = []
        for place in places:
            position = geocoder.bing(place, key=BING_KEY)                   
            if position.json:
                #get the coordinates of the bbox
                lat_min = position.json['bbox']['southwest'][0]
                lat_max = position.json['bbox']['northeast'][0]
                lon_min = position.json['bbox']['southwest'][1]
                lon_max = position.json['bbox']['northeast'][1]
            
                bbox = ((lat_min, lat_max), (lon_min, lon_max))
                coord_bbox.append(bbox)
            
                #get coordinate of the place
                lat_lon = (position.json['lat'], position.json['lng'])
                coord_couples.append(lat_lon)
            
        if len(coord_couples) > 1:
            candidate_places, coord_couples = prefilteringCoordinates(places, coord_couples, coord_bbox)
        
            if len(candidate_places) > 1:
                
                if heuristic == 1:
                    #euristica1: selezione del primo luogo rilevato
                    coordinate = coord_couples[0]
                elif heuristic == 2:
                    place = " ".join(candidate_places)
                    position = geocoder.bing(place, key=BING_KEY)
                    lat_lon = (position.json['lat'], position.json['lng'])
                    coordinate = lat_lon
            elif len(candidate_places):
                coordinate = coord_couples[0]
                
        elif len(coord_couples) == 1:
            coordinate = coord_couples[0]
    
    
    if coordinate is None:
        #hastag = getFirstHastag(tweet)
        hastags = getHastags(tweet)
        for i in range(0, len(hastags)):
            place = hastags[i].replace("#", "")
            position = geocoder.bing(place, key=BING_KEY) 
            if position.json:
                lat_lon = (position.json['lat'], position.json['lng'])
                coordinate = lat_lon
                break
              
    return coordinate


def accuracyScore(coords_gold, coords):
    counter = 0
    for i, coord_pred in enumerate(coords):
        coord_gold = coords_gold[i]
        if not isinstance(coord_pred, type(None)):
            distance = round(geopy.distance.geodesic(coord_gold, coord_pred).km, 3)
            #tolleranza di 10km
            if distance <= 0:
                counter += 1
    return counter / (i+1)
        
   

    
    """locator = Nominatim(user_agent ="myGeocoder")
    location = locator.geocode(place)
    if location:
        print("{}, {}".format(location.latitude, location.longitude))"""
    
    """gmaps = googlemaps.Client(key = 'YOUR KEY')
    coords = gmaps.geocode(place)
    lat = coords[0]["geometry"]["location"]["lat"]
    lon = coords[0]["geometry"]["location"]["lng"]"""         