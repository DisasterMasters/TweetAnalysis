import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import itertools


tweets = []
labels_list = []

file = open("random_100s_F.csv") 
csv_read = csv.reader(file)
csv_read.next()
tmp_list = []
for row in csv_read:
	label = int(row[0].split('.')[0])
	tweet = row[1]
	tweets.append(tweet)
	labels_list.append(label)


file = open("old/twits.txt", "r")
w = file.read()
test_data = w.split("\t")


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

outfile = open("results/one1000more.csv", "w")
writer = csv.writer(outfile)
writer.writerow(['Tweet', 'Category'])

def writeFile(category):

	cat = probs.sort_values(category, ascending=False).head(1000).index
	for i, v in enumerate(test_data):
		if i in cat:
			writer.writerow([v, category])

for i in range(0, 12):
	writeFile(i)
