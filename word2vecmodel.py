import pandas as pd

train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter ="\t", quoting=3)
test = pd.read_csv("testData.tsv", header = 0, delimiter ="\t", quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header = 0, delimiter ="\t", quoting=3)

print "%d labeled data,%d test data, %d unlabeled data"%(train["review"].size,test["review"].size,unlabeled_train["review"].size)
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist(review,remove_stopwords = False):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-z]"," ",review_text)#check with both,with or without
    words = review_text.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        
        
    return  words
#downloading the punkt tokenizer for sentence splitting     
import nltk.data
nltk.download() 


#load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#defining the function to convert review into parsed sentences

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []

print "parsing sentences from training set"

for review in train["review"]:
	sentences += review_to_sentences(review,tokenizer)
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print "parsing done"    
print len(sentences)

