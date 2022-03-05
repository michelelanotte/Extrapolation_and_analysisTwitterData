from utils import cleaningForNE, getNE, re, getCoordFromPlace, accuracyScore
import csv 

"""
This method is based on spacy to perform Named Entities Recognition
"""
def computeNamedPlaces(tweet):
    filters = ["PERSON", "TIME", "DATE", "ORDINAL", "MONEY", "WORK_OF_ART", "CARDINAL", "PRODUCT", "PERCENT", "EVENT"]
    entities = getNE(tweet, filters)       
    return entities


"""
This method excludes incorrect named entities from name_entities_list. 
Only one location will be returned for the current tweet. 
"""
def filteringNE(i, name_entities_list):
    ne_filtered = []
    for ne in name_entities_list:
        pattern_re = "\\x80.*|\\x98.*|\\x81.*|.\\x9f.*|.\\x9c.*|.\\x9d.*|.\\x99.*|@"
        ne = re.sub(pattern_re, ' ', ne).strip()
        if len(ne) > 1:
            ne_filtered.append(ne.lower())
    
    #remove duplicate
    res = []
    for elem in ne_filtered:
        if elem not in res:
            res.append(elem)
    return res
            

def main():
    with open("dataset/tweets_data.tsv", "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = '\t')
        tweets = []
        coords_gold = []
        for entity in csv_reader:
            tweets.append(entity[0])
            coords_gold.append((float(entity[1]) , float(entity[2])))
    
    coords = []
    heuristic = 3
    
    """for i, tweet in enumerate(tweets):
        print(i)
        #The removal of punctuation, double characters, url and tag was not considered, as there is an overall 
        #decrease in performance (in precision and recall)
        tweet_cleaned = cleaningForNE(tweet)
        #named_entities_list is a list of named entities detected in current tweet
        named_entities_list = computeNamedPlaces(tweet_cleaned)
        named_entities_filtered = filteringNE(i, named_entities_list)
        
        coords.append(getCoordFromPlace(tweet, named_entities_filtered, heuristic)) 
    
    with open("result_" + str(heuristic) + ".tsv", 'w') as tsvfile:
        for coord in coords:
            if not isinstance(coord, type(None)):
                tsvfile.write(str(coord[0]) + "\t" + str(coord[1]) + "\n")
            else:
                tsvfile.write("-\n")"""
    
    
    #compute score
    coords = []
    with open("result_" + str(heuristic) + ".tsv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = '\t')
    
        for elem in csv_reader:
            if elem[0] != "-":
                coords.append((float(elem[0]), float(elem[1])))
            else:
                coords.append(None)
    
    print("Accuracy: ", round(accuracyScore(coords_gold, coords), 2))

main()