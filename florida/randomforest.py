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


tweets = []
labels_list = []

tweetlabel_dict = {}

file = open("training_data/coding_7_3.csv") 
csv_read = csv.reader(file)
header = csv_read.next()
for row in csv_read:
	if row[1] != '':
		label = row[1]
		if '.' in label:
			label = re.match(r'^(.*?)\..*', label).group(1)
		label = int(label)
		tweet = row[0]
		if label != 17:
			if label not in tweetlabel_dict:
				tweetlabel_dict[label] = [tweet]
			else:
				tweetlabel_dict[label].append(tweet)

tweets = []
labels_list = []

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
'''
file = open("tmp.csv", "w")
csv_w = csv.writer(file)
csv_w.writerow(["Tweet", "Label"])
count_dict = {}
for l, t in itertools.izip(labels_list, tweets):
	if l not in count_dict:
		count_dict[l] = 1
	else:
		count_dict[l] += 1
	csv_w.writerow([t, str(l)])

print count_dict
'''


date_dict = {}

test_data = []

typeoffile = sys.argv[1] #media or utility
file_name = 'dates'
if typeoffile == 'media':
	file_name = 'm_dates'

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
				start = datetime(year=2017, month=9, day=1)
				end = datetime(year=2017, month=9, day=30)
				if start < date < end:
					test_data.append([t[0], date, t[2]])

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(tweets).toarray()
labels = labels_list

X_train, X_test, y_train, y_test = train_test_split(tweets, labels, random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = RandomForestClassifier().fit(X_train_tfidf, y_train)


outfile = open("results/" + typeoffile + "_supervised_rf.csv", "w")
writer = csv.writer(outfile)
writer.writerow(['Tweet', 'Category', 'Date', 'Permalink'])


for t in test_data:
	vect = count_vect.transform([t[0]])
	prediction = clf.predict(vect)
	writer.writerow([t[0], int(prediction), t[1].strftime('%m/%d/%Y'), t[2]])
	
#probs = pd.DataFrame(clf.predict_proba(tweet_test))

'''
def writeFile(category):

	
	cat = probs.sort_values(category, ascending=False).index
	for i, v in enumerate(test_data):
		if i in cat:
			writer.writerow([v, category])

for i in range(0, list_end):
	writeFile(i)'''
