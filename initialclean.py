
import pandas as pd    
train = pd.read_csv("labeledTrainData.tsv", header=0, \
delimiter="\t", quoting=3)
#print train.shape
#print train.columns.values
#print train["review"][0]
from bs4 import BeautifulSoup 
example1 = BeautifulSoup(train["review"][0])
#print train["review"][0]
#print example1.get_text()# gives the text of the review
 
 
#dealing with punctuations,numbers and stopwords
import re 
#using regular expressions to do a find and replace all the non alphabetical elements

letters_only = re.sub("[^a-zA-Z]"," ",example1.get_text())
#print letters_only
lower_case = letters_only.lower()#convert into lower case
words = lower_case.split()#split into words

import nltk
from nltk.corpus import stopwords#import stopwords list
#print stopwords.words("english")
#removing stop words from the words
words =[w for w in words if not w in stopwords.words("english")];
print words
