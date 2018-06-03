from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import io
import csv 
import os


def display_topics(model, feature_names, no_top_words):
	nmf = open("results/nmf.txt", "w")
	for topic_idx, topic in enumerate(model.components_):
		nmf.write("Topic %d:" % (topic_idx))
		nmf.write(" ".join([feature_names[i]
                      for i in topic.argsort()[:-no_top_words - 1:-1]]))

documents = []

with open("twits.txt") as file:
	for line in file:
		documents = line.split("\t") 
		

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)


no_top_words = 10
print "\n********Non-negative Matrix Factorization (NMF)***********\n"
display_topics(nmf, tfidf_feature_names, no_top_words)
				


