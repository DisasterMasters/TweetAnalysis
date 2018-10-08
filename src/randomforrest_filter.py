import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def build_model(verbose=False):
    if (verbose):
        print("Loading Training Data")

    data_source = r"/home/manny/PycharmProjects/TweetAnalysis/DATA/T-06-Hurricane_Sandy_Labeled/2012_Sandy_Hurricane-ontopic_offtopic.csv"

    train = pd.read_csv(data_source, names=['tweet id', 'tweet', 'label'])

    train['label'] = train['label'].map({'on-topic': int(1), 'off-topic': int(0)})
    train['label'] = train['label'].fillna(value=0)
    y = train['label']
    x = train['tweet'].fillna(" ")
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

        clf = RandomForestClassifier(verbose=2,  max_features=.2, n_estimators=12, min_samples_leaf=3)
    else:

        clf = RandomForestClassifier(verbose=0, max_features=.2, n_estimators=12, min_samples_leaf=3)

    clf.fit(x, y)

    sample_source = r"/home/manny/PycharmProjects/TweetAnalysis/src/Sample.csv"

    test = pd.read_csv(sample_source)

    test['Manual'] = test['Manual'].fillna(value=0)
    y = test['Manual']
    x = test['Tweet'].fillna(" ")
    if (verbose):
        print("Loading Validation set")

    x = tfidf.transform(x).toarray()

    print(clf.score(x, y))
    return tfidf, clf


tfidf, clf = build_model(True)

while (True):
    print(clf.predict(tfidf.transform([input("Input test:   ")]).toarray())[0])
