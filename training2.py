# -*- coding: utf-8 -*-
"""
Created on Sun Dec 06 20:45:19 2015

@author: ADMIN
"""

# Load the model that we created in Part 2
from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")
print model.syn0.shape


