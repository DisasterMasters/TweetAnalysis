import pandas as pd
from sklearn.ensemble import RandomForestClassifier

training_data_path = r"/home/manny/PycharmProjects/TweetAnalysis/florida/training_data"
results_path = r"/home/manny/PycharmProjects/TweetAnalysis/florida/results"

tweets = []
labels_list = []

tweetlabel_dict = {}

from sklearn.model_selection import train_test_split

# random

from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from nltk.stem.wordnet import WordNetLemmatizer
import re

lmtzr = WordNetLemmatizer()
w = re.compile("\w+", re.I)


def label_sentences(df):
    labeled_sentences = []
    for index, datapoint in df.iterrows():
        tokenized_words = re.findall(w, datapoint["Tweet"].lower())
        labeled_sentences.append(LabeledSentence(words=tokenized_words, tags=['SENT_%s' % index]))
    return labeled_sentences


def train_doc2vec_model(labeled_sentences):
    model = Doc2Vec(alpha=0.025, min_alpha=0.025)
    model.build_vocab(labeled_sentences)
    for epoch in range(10):
        model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    return model


def vectorize_comments(df, d2v_model):
    y = []
    comments = []
    for i in range(0, df.shape[0]):
        label = 'SENT_%s' % i
        comments.append(d2v_model.docvecs[label])
    df['vectorized_comments'] = comments

    return df


train = pd.read_csv(
    "/home/manny/PycharmProjects/TweetAnalysis/florida/training_data/supervised_data/utility/combined.csv")

# Removing Category 7 from the frame
train = train[train['Manual Coding'] != 7]

sen = label_sentences(train)
model = train_doc2vec_model(sen)
# print(sen)
df = vectorize_comments(train, model)
# print(df.head(2))

xtrain, xtest, ytrain, ytest = train_test_split(df['vectorized_comments'].T.tolist(), df['Manual Coding'])

clf = RandomForestClassifier(max_features='sqrt', n_estimators=40, n_jobs=-1).fit(xtrain, ytrain)

print(clf.score(xtest, ytest))
