import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import itertools
from datetime import datetime
import re


tweets = []
labels_list = []

file = open("training_data/random_100s_F.csv") 
csv_read = csv.reader(file)
csv_read.next()
tmp_list = []
for row in csv_read:
	label = row[0]
	if '.' in label:
		label = re.match(r'^(.*?)\..*', label).group(1)
	label = int(label)
	tweet = row[1]
	tweets.append(tweet)
	labels_list.append(label)

list_end = labels_list[-1] + 1

test_data = []
file = open("training_data/dates.txt", "r")
w = file.read()
test = w.split("\t")
for t in test:
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
				test_data.append(t[0])


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(tweets).toarray()
labels = labels_list

X_train, X_test, y_train, y_test = train_test_split(tweets, labels, random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

tweet_test = count_vect.transform(test_data)

probs = pd.DataFrame(clf.predict_proba(tweet_test))

outfile = open("results/utility_supervised.csv", "w")
writer = csv.writer(outfile)
writer.writerow(['Tweet', 'Category'])

def writeFile(category):

	cat = probs.sort_values(category, ascending=False).index
	for i, v in enumerate(test_data):
		if i in cat:
			writer.writerow([v, category])

for i in range(0, list_end):
	writeFile(i)
