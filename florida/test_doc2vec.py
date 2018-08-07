import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import itertools
from datetime import datetime
import re
import math
import random
from itertools import repeat
import sys
import os
from gensim import utils
from gensim.models import Doc2Vec
import gensim
import numpy as np
import nltk
import io
from sklearn.model_selection import KFold
from itertools import izip

#from medium.com (mostly)
class LabeledLineSentence(object):
	
	def __init__(self, doc_list, labels_list):
		self.labels_list = labels_list
		self.doc_list = doc_list

	def __iter__(self):
		for t, l in itertools.izip(self.doc_list, self.labels_list):
			t = t.decode('utf-8')
			t = nltk.word_tokenize(t)
			yield gensim.models.doc2vec.LabeledSentence(t, [l])


tweets = []
labels_list = []


#need this here since the training data for different categories is different
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
				if label != 17: #ignore label 17 ('not sure')
					labels_list.append(label)
					tweets.append(tweet)


X_train, X_test, y_train, y_test = train_test_split(tweets, labels_list, test_size=0.33, random_state=42)

#put into proper format
training_data = LabeledLineSentence(X_train, y_train)

#build the doc2vec model
model = Doc2Vec(vector_size=50, min_count=3, dm=1)
model.build_vocab(training_data)
model.train(training_data, total_examples=model.corpus_count, epochs=200)

counter = 0.0
correct = 0.0

for x, y in itertools.izip(X_test, y_test):
	split = x.decode('utf-8')
        split = nltk.word_tokenize(split)
        vect = model.infer_vector(split)
        #get cosine similarity, keep highest score
        result = model.docvecs.most_similar(positive=[vect], topn=1)
        prediction = result[0][0]
	counter += 1.0
	if int(prediction) == int(y):
		correct += 1.0

percent_correct = (correct/counter) * 100
print "Percent correct: " + str(percent_correct) + "%"


	


	
