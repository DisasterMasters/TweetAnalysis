
import csv
import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
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
        header = next(csv_read)
        for row in csv_read:
            tweet = row[0]
            tweets.append(tweet)


train = pd.read_csv('/home/manny/PycharmProjects/TweetAnalysis/DATA/T-06-Hurricane_Sandy_Labeled/2012_Sandy_Hurricane-ontopic_offtopic.csv', names = ['tweet id', 'tweet' , 'label'])




train['label'] = train['label'].map({'on-topic': int(1), 'off-topic': int(0)})
train['label'] = train['label'].fillna(value=0)
print(train.label.unique())

y = train['label']
X = train['tweet']

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')

tfidf = tfidf.fit(tweets)
tfidf = tfidf.fit(X)


X_vectorized = tfidf.transform(X)

test = tfidf.transform("Hurricane")

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y)


# input("K")
model = RandomForestClassifier()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("score: " + str(score))

# save the model to disk
filename = '/home/manny/PycharmProjects/TweetAnalysis/florida/results/nn_cleaning/randomforest_model/finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
results = model.predict(test)
print(results)