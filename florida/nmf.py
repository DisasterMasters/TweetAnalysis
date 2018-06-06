from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import io
import csv
import os
import sys

def display_topics(model, feature_names, no_top_words):
    nmf = open("results/topics.txt", "w")
    for topic_idx, topic in enumerate(model.components_):
        nmf.write("Topic %d:" % (topic_idx))
        nmf.write("\n")
        nmf.write(" ".join([feature_names[i]
                      for i in topic.argsort()[:-no_top_words - 1:-1]]))
        nmf.write("\n")

documents = []

with open("training_data/dates.txt") as file:
    for line in file:
        docs = line.split("\t")
        for d in docs:
			t = d.split('+++')[0]
			documents.append(t)

count = open("len_ut.txt", "w")
count.write(str(len(documents)))


no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvda').fit(tfidf)


no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)


nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvda')
W = nmf.fit_transform(tfidf) #column

#look through indices for W
#W corresponds to index in documents
#number of W corresponds to topic in document
#numpy.argsort()

file_w = open("results/W_indices.txt", "w")

for w in W:
        file_w.write(str(w))
