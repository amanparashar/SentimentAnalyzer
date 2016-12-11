
# Load the model that we created in Part 2
from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")
#print model.syn0.shape

import numpy as np

def makeFeatureVec(words,model,num_features):
    #function to average all of the word vectors in a given paragraph
    featureVec = np.zeros((num_features,),dtype="float32")
    
    nwords =0
    #index2word is a list that contains the names of the words in the model's vocabulary.
    index2word_set = set(model.index2word)
    #
    #looping over each review 
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
            
def getAvgFeatureVecs(reviews,model,num_features):
    # given a set of reviews(each one a list of words),calculate 
    #the average feature vector of each one and return a 2d numpy array
    
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype ="float32")        
    
    for review in reviews:
        if counter%1000. == .0 :
            print "Review %d of %d" % (counter,len(reviews))
        reviewFeatureVecs[counter] = makeFeatureVec(review,model,num_features)
        counter = counter+1.
return reviewFeatureVecs

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review,remove_stopwords = True))
trainDataVecs = getAvgFeatures(clean_train_reviews,model,num_features)       

print "Creating average feature vecs for test reviews"

clean_test_reviews = []

for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords = True))
testDataVecs = getAvgFeatureVec(clean_test_reviews,model,num_features)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimator = 100)

results = forest.predict(testDataVecs)

output1 = pd.DataFrame(data = {"review":test["review"],"sentiment":result})
output2 = pd.DataFrame(data = {"id":test["id"],"sentiment":result})

output2.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
    
    
        