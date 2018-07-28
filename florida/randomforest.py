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

tweetlabel_dict = {}

typeoffile = sys.argv[1] #media or utility

included = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] 

file_name = ''
if typeoffile == 'utility':
	file_name = "dates"
	included = [1, 4, 8, 9, 10, 11, 12, 14, 15] 
elif typeoffile == 'media':
	file_name = 'm_dates'



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
				if label in included:
					#put into dictionary so we can count the tweets per label
					if label not in tweetlabel_dict:
						tweetlabel_dict[label] = [tweet]
					else:
						tweetlabel_dict[label].append(tweet)

tweets = []
labels_list = []

#balancing out the training data (~500 in each category)
for k, v in tweetlabel_dict.iteritems():

	if len(v) > 500:
		sample = random.sample(v, 500)
		for s in sample:
			tweets.append(s)
		labels_list.extend(repeat(k, 500))

	elif len(v) < 500:
		iters = math.ceil(500.0/len(v))
		for i in range (0, int(iters)):
			for vals in v:
				tweets.append(vals)
				labels_list.append(k)	

date_dict = {}
test_data = []


#open test data obtained from media/utility file, parse
file = open("training_data/" + file_name + ".txt", "r")
w = file.read()
test = w.split("\t")
for t in test:
	if t != '\n':
		t = t.split('~+&$!sep779++')
		if t != ['']:
			if t[1]:
				t[1] = t[1].strip()
				date = ''
				if '/' in t[1]:
					date = datetime.strptime(t[1], '%m/%d/%Y')
				elif '-' in t[1]:
					date = datetime.strptime(t[1], '%Y-%m-%d')
				#make sure only data from september is included
				start = datetime(year=2017, month=9, day=1)
				end = datetime(year=2017, month=9, day=30)
				if start < date < end:
					test_data.append([t[0], date, t[2]])

#fit tf-idf with the training data
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(tweets).toarray()
labels = labels_list

#train the model 
X_train, X_test, y_train, y_test = train_test_split(tweets, labels, random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = RandomForestClassifier().fit(X_train_tfidf, y_train)

#open predictions file for writing
outfile = open("results/" + typeoffile + "_supervised_rf.csv", "w")
writer = csv.writer(outfile)
writer.writerow(['Tweet', 'Category', 'Date', 'Permalink'])

#make prediction for each tweet, write to file
for t in test_data:
	vect = count_vect.transform([t[0]])
	prediction = clf.predict(vect)
	if prediction in included:
		#writing tweet, prediction, date, and permalink
		writer.writerow([t[0], int(prediction), t[1].strftime('%m/%d/%Y'), t[2]])
	

