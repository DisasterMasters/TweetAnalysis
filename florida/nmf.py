from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import io
import csv
import os
import sys

#function to display topics from nmf
def display_topics(model, feature_names, no_top_words, typeoffile):
    nmf = open("results/" + typeoffile + "topics.txt", "w")
    for topic_idx, topic in enumerate(model.components_):
        nmf.write("Topic %d:" % (topic_idx))
        nmf.write("\n")
        nmf.write(" ".join([feature_names[i]
                      for i in topic.argsort()[:-no_top_words - 1:-1]]))
        nmf.write("\n")

documents = []

#select which file to run nmf on
typeoffile = sys.argv[1]

if typeoffile == 'utility':
        outfile = 'training_data/utility_data.txt'
if typeoffile == 'media':
        outfile = 'training_data/media_data.txt'
if typeoffile == 'nonprofit'
        outfile = 'training_data/nonprofit_data.txt'
if typeoffile == 'gov'
        outfile = 'training_data/gov_data.txt'

#splitting and opening file
with open(outfile) as file:
    for line in file:
        docs = line.split("\t")
        for d in docs:
			t = d.split('~+&$!sep779++')[0]
			if t != '':
				documents.append(t)


no_features = 500

#put into tf-idf 
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

#select number of topics you want to see
no_topics = 5

#run nmf, save result for weights
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvda').fit(tfidf)

#display 10 words from nmf
no_top_words = 10
#call function to display topics
display_topics(nmf, tfidf_feature_names, no_top_words, typeoffile)

#run nmf again to obtain weights column for tweet lookup
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvda')
W = nmf.fit_transform(tfidf) #column

#open file to write weights column
file_w = open("results/W_indices" + typeoffile + ".txt", "w")
for w in W:
        file_w.write(str(w))
