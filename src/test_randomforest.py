import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import itertools
from datetime import datetime
import re
import math
import random
from itertools import repeat
import sys
import os


tweets = []
labels_list = []


typeoffile = sys.argv[1]

file_name = ''

if typeoffile == 'utility':
        file_name = 'training_data/utility_data.txt'
if typeoffile == 'media':
        file_name = 'training_data/media_data.txt'
if typeoffile == 'nonprofit':
        file_name = 'training_data/nonprofit_data.txt'
if typeoffile == 'gov':
        file_name = 'training_data/gov_data.txt'



#reading in training data
for dirs, subdirs, files in os.walk("training_data/supervised_data/" + typeoffile):  #all data for supervised learning should be put in this directory
	for fname in files:
		file = open(dirs + "/" + fname, "r")
		csv_read = csv.reader(file)
		header = csv_read.next()
		for row in csv_read:
			if row[1] != '' and len(row[1].split(' ')) == 1: #try to keep tweet as first entry and label as second
				label = row[1]
				if '.' in label: #get rid of decimal labeling if there is any
					label = re.match(r'^(.*?)\..*', label).group(1)
				label = int(label)
				tweet = row[0]
				if label != 17:
					labels_list.append(label)
					tweets.append(tweet)



#fit tf-idf with the training data
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(tweets).toarray()
labels = labels_list

#train the model and split into test-train
X_train, X_test, y_train, y_test = train_test_split(tweets, labels, random_state=0)
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
clf = RandomForestClassifier().fit(X_tfidf, y_train)

#fit the test data
X_test_count = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_count)

#print score (0.404 as of 07/31/2018)
print clf.score(X_test_tfidf, y_test)


