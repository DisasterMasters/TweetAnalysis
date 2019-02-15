
# coding: utf-8

# In[4]:


from __future__ import print_function

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF


import csv
import os
import sys
import pandas as pd


n_features = 500
n_components = 2 #this is the number of topics
n_top_words = 10

# stop words is a dictionary of about 300 words. These are additional "words" 
# that are basically noise/nonsense and don't contribute anything, and are thus left out
# most of them are from hyperlinks
my_stop_words = text.ENGLISH_STOP_WORDS.union(["http", "bit", "ly", "pic", "twitter", "www", "com", "https", "rt", "hurricane", "irma"])

#print topics
def print_topics(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def save_topics(model, feature_names, n_top_words, n_components):
    f = open(str(n_components) + "_topics.txt", "w")
    for topic_idx, topic in enumerate(model.components_):
        f.write("Topic %d:" % (topic_idx))
        f.write("\n")
        f.write(" ".join([feature_names[i]
                         for i in topic.argsort()[:-n_top_words -1:-1]]))
        f.write("\n")

corpus = pd.read_csv('corpus.csv')
corpus = corpus.applymap(str)
corpus = corpus['Tweet'].fillna(" ")

tweets = pd.read_csv("Tweetcat22.csv")



# Use tf-idf features for NMF.
print("Extracting tf-idf features for NMF...")
tfidf_vectorizer = TfidfVectorizer(max_df=.50, min_df=5,
                                   max_features=n_features,
                                   stop_words=set(my_stop_words), ngram_range=(1,3))
#fit_transform??
print("Fitting corpus...")
tfidf_vectorizer.fit(corpus)

print("Transforming tweets...")
tfidf = tfidf_vectorizer.transform(tweets["Tweet"].fillna(" "))


# Fit the NMF model
print("Fitting NMF model...")
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5, init = 'nndsvda').fit(tfidf)


print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_topics(nmf, tfidf_feature_names, n_top_words)

# Fit the NMF model
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)


print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_topics(nmf, tfidf_feature_names, n_top_words)


# In[2]:


# Run NMF again to obtain weights column for tweet lookup
print('Calculating weights...')
W = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvda').fit_transform(tfidf)

# write weights to a file
file = open("weight_indices_" + str(n_components) + ".txt", "w")
for w in W:
    file.write(str(w))
    
print("File weight_indices_" + str(n_components) + ".txt created")


# In[3]:


save_topics(nmf, tfidf_feature_names, n_top_words, n_components)
print(str(n_components) + "_topics.txt created")

