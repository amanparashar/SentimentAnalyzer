

import pandas as pd    
import numpy 
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
train = pd.read_csv("labeledTrainData.tsv", header=0, \
delimiter="\t", quoting=3)
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
#print train.shape
#print train.columns.values
#print train["review"][0]
from bs4 import BeautifulSoup 
#example1 = BeautifulSoup(train["review"][0])
#print train["review"][0]
#print example1.get_text()# gives the text of the review
 
 
#dealing with punctuations,numbers and stopwords
import re 
#using regular expressions to do a find and replace all the non alphabetical elements

#letters_only = re.sub("[^a-zA-Z]"," ",example1.get_text())
#print letters_only
#lower_case = letters_only.lower()#convert into lower case
#words = lower_case.split()#split into words

import nltk
from nltk.corpus import stopwords#import stopwords list
#print stopwords.words("english")
#removing stop words from the words
#words =[w for w in words if not w in stopwords.words("english")];
#print words;
def review_to_words(raw_review):
     #function to convert a raw movie review to words
     #the input is a single string (a raw movie review)
     #output will be a single string(a preprocessed movie review)
     #1. remove HTML tags
     review_text = BeautifulSoup(raw_review).get_text()
     #2.remove non letters
     letters_only = re.sub("[^a-zA-Z]"," ",review_text)
     
     #3.Convert to lower case,split into individual words
     words = letters_only.lower().split()
     
     #4. converting the stopwords into a set
     stops = set(stopwords.words("english"))
     meaningful_words = [w for w in words if not w in stops]
     
     return (" ".join(meaningful_words))
     
#clean_review = review_to_words(train["review"][0])
#print clean_review    
num_reviews = train["review"].size

clean_train_reviews = []
print "Cleaning and parsing the training set movie reviews....\n"
for i in xrange(0,num_reviews):
    clean_train_reviews.append(review_to_words(train["review"][i]))
    if((i+1)%1000 == 0):
        print "review %d of %d\n"%(i+1,num_reviews)
    
print "cleaning of the set complete.."

print "creating the bag of words..."

from sklearn.feature_extraction.text import CountVectorizer
#initializing the countvectorizer which is the bag of words tool for sklearn
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,  preprocessor = None, stop_words = None, max_features = 5000) 
y = train["sentiment"]  


import numpy as np

#fit_transform() 1.) learns the vocabulary
#                2.) fit a count model to the input string,convert input string into a feature vector

train_data_features = vectorizer.fit_transform(clean_train_reviews)
np.array(train_data_features)

num_reviews = len(test["review"])
clean_test_reviews = [] 

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print "Review %d of %d\n" % (i+1, num_reviews)
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
np.asarray(test_data_features)


print train_data_features.size
vectorizer.fit_transform(clean_train_reviews)
#print vectorizer.vocabulary_
X = train_data_features
# Numpy arrays are easy to work with, so convert the result to an 
# array
#train_data_features = train_data_features.toarray()
#vocab = vectorizer.get_feature_names()

model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0,class_weight=None, random_state=None)
                         

print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model,X, y, cv=20, scoring='roc_auc'))

print "Retrain on all training data, predicting test labels...\n"
model.fit(X,y)
result = model.predict_proba(test_data_features)[:,1]
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Veectors.csv", index=False, quoting=3 )    
for i in xrange(0,5) :
 print "the prediction for the review %s\n"%test["review"][i]
 print result[i]
                     
print "model complete!"