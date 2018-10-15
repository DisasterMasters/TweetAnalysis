import glob

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

training_data_path = r"/home/manny/PycharmProjects/TweetAnalysis/florida/training_data"
results_path = r"/home/manny/PycharmProjects/TweetAnalysis/florida/results"

tweets = []
labels_list = []

tweetlabel_dict = {}

# pick file: media, utility, gov, or nonprofit
# typeoffile = sys.argv[1]
typeoffile = 'utility'

included = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]

file_name = ''

if typeoffile == 'utility':
    file_name = training_data_path + '/utility_data_(not_utility_source).txt'
    included = [1, 4, 8, 9, 10, 11, 12, 14, 15]
if typeoffile == 'media':
    file_name = training_data_path + '/media_data.txt'
    included = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
if typeoffile == 'nonprofit':
    file_name = training_data_path + '/nonprofit_data.txt'
if typeoffile == 'gov':
    file_name = training_data_path + '/gov_data.txt'

# # reading in testing data
# test_data = []
# # open test data obtained from media/utility file, parse
# file = open(file_name, "r")
# w = file.read()
# test = w.split("\t")
# for t in test:
#     if t != '\n':
#         t = t.split('~+&$!sep779++')
#         if t != ['']:
#             test_data.append([t[0], t[1], t[2]])

test = pd.read_csv(file_name)
# test.columns = ["Tweet", "Date", "Link"]

# reading in training data
frame = pd.DataFrame()
list_ = []

files = glob.glob(training_data_path + "/supervised_data/" + typeoffile + "/*.csv")
for fname in files:
    df = pd.read_csv(fname, index_col=None, header=0)
    list_.append(df)

frame = pd.concat(list_, sort=False)

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

test.to_csv('/home/manny/PycharmProjects/TweetAnalysis/florida/results/utility_supervised_rf.csv')
