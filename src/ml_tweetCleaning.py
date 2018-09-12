import csv
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

training_data_path = r"/home/manny/PycharmProjects/TweetAnalysis/florida/training_data/"

tweets = []
labels_list = []

#reading in training data
for dirs, subdirs, files in os.walk(training_data_path):  #all data for supervised learning should be put in this directory
    for fname in files:
        file = open(dirs + "/" + fname, "r")
        # print(fname)
        csv_read = csv.reader(file)
        header = csv_read.next()
        for row in csv_read:
            tweet = row[0]
            tweets.append(tweet)


train = pd.read_csv('/home/manny/PycharmProjects/TweetAnalysis/DATA/T-06-Hurricane_Sandy_Labeled/2012_Sandy_Hurricane-ontopic_offtopic.csv', names = ['tweet id', 'tweet' , 'label'])

y = train['label']
X = train['tweet']


train['label'] = train['label'].map({'on-topic': 1, 'off-topic': 0})

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')

tfidf = tfidf.fit(tweets)
tfidf = tfidf.fit(X)


X_vectorized = tfidf.transform(X)


sum = 0
for i in range(5):

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y)


    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    sum += score
    print("test score: " + str(score))

print("=================" + "\n" +  "Average Score: " + str(sum /5))
