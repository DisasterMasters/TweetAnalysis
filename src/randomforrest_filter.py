import glob
from collections import Counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


def build_model(verbose=False):
    if (verbose):
        print("Loading Training Data")

    data_source = r"/home/manny/PycharmProjects/TweetAnalysis/DATA/T-06-Hurricane_Sandy_Labeled/"

    training_list = []
    files = glob.glob(data_source + "/*.csv")
    for fname in files:
        df = pd.read_csv(fname, index_col=None, header=0)
        training_list.append(df)

    train = pd.concat(training_list, sort=False, ignore_index=True)
    training_list = []

    # print(Counter(train['Label']))

    train['Label'] = train['Label'].fillna(value=0)
    y = train['Label']
    x = train['Tweet'].fillna(" ")

    if (verbose):
        print("Loading Corpus")
    corpus = pd.read_csv('corpus.csv')
    corpus = corpus.applymap(str)
    corpus = corpus['Tweet'].fillna("  ")

    # fit tf-idf with the training data
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')
    if (verbose):
        print("Fitting Tfidf")

    tfidf.fit(corpus)
    tfidf.fit(x)

    x = tfidf.transform(x).toarray()
    if (verbose):
        print("Training Model")

    # train the model and split into test-train
    # X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

    if (verbose):

        clf = RandomForestClassifier(verbose=2, max_features=.2, n_estimators=12, min_samples_leaf=3, n_jobs=-1)
    else:

        clf = RandomForestClassifier(verbose=0, max_features=.2, n_estimators=12, min_samples_leaf=3, n_jobs=-1)

    clf.fit(x, y)

    # sample_source = r"/home/manny/PycharmProjects/TweetAnalysis/DATA/T-06-Hurricane_Sandy_Labeled/Manually_labelled_data_2.csv"
    #
    # test = pd.read_csv(sample_source)
    #
    # test['Label'] = test['Label'].fillna(value=0)
    # # y = test['Label']
    # x = test['Tweet'].fillna(" ")
    # if (verbose):
    #     print("Loading Validation set")
    #
    # x = tfidf.transform(x).toarray()
    #
    # print(clf.score(x, y))
    return tfidf, clf
#

tfidf, clf = build_model(True)

while (True):
    print(clf.predict(tfidf.transform([input("Input test:   ")]).toarray())[0])
