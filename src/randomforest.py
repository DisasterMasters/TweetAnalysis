import glob

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import imblearn

training_data_path = r"/home/manny/PycharmProjects/TweetAnalysis/florida/training_data"
results_path = r"/home/manny/PycharmProjects/TweetAnalysis/florida/results"

tweets = []
labels_list = []

tweetlabel_dict = {}

# pick file: media, utility, gov, or nonprofit
# typeoffile = sys.argv[1]
typeoffile = 'utility'

file_name = ''

if typeoffile == 'utility':
    file_name = training_data_path + '/utility_data_(not_utility_source).txt'
    train_name = training_data_path + "/supervised_data/utility/combined.csv"
    included = [1, 4, 8, 9, 10, 11, 12, 14, 15]
if typeoffile == 'media':
    file_name = training_data_path + '/media_data.txt'
    included = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
if typeoffile == 'nonprofit':
    file_name = training_data_path + '/nonprofit_data.txt'
if typeoffile == 'gov':
    file_name = training_data_path + '/gov_data.txt'

test = pd.read_csv(file_name)
# test.columns = ["Tweet", "Date", "Link"]

# reading in training data

train = pd.read_csv(train_name)
# print(list(frame.columns.values))


# Removing Category 7 from the frame
frame = frame[frame['Manual Coding'] != 7]

# Loading Corpus to train tfidf on
corpus = pd.read_csv('corpus.csv')
corpus = corpus.applymap(str)
corpus = corpus['Tweet'].fillna("  ")

# fit tf-idf with the training data
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')

# Training tfidf on corpus
# tfidf.fit(frame['Tweet'])
tfidf.fit(corpus)
# tfidf.fit(test['Tweet'])

# train the model

x = tfidf.transform(frame['Tweet'])
y = frame['Manual Coding']
xtest = tfidf.transform(test['Tweet'])

# xtrain, xtest, ytrain, ytest = train_test_split(x, y)

clf = RandomForestClassifier(max_features='sqrt', n_estimators=40, n_jobs=-1).fit(x, y)
# print( clf.score(xtest, ytest))

test['Category'] = clf.predict(xtest)

test.drop(test.iloc[:, 0:1], inplace=True, axis=1)

test.to_csv('/home/manny/PycharmProjects/TweetAnalysis/florida/results/utility_supervised_rf.csv')
