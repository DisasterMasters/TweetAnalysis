import collections
import re
import gensim
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Score(collections.namedtuple("Score", "overall positive negative neutral")):
    def __add__(self, rhs):
        return Score(
            overall = self.overall + rhs.overall,
            positive = self.positive + rhs.positive,
            negative = self.negative + rhs.negative,
            neutral = self.neutral + rhs.neutral,
        )

class doc2vec:

    def __init__(self, df, X, Y, build=False):
        self.w = re.compile("\w+", re.I)
        if 'basestring' not in globals():
            basestring = str

        # Hyperparameters : https://arxiv.org/pdf/1607.05368.pdf
        self.vector_size = 300
        self.window_size = 15
        self.min_count = 1
        self.sampling_threshold = 1e-4
        self.negative_size = 5
        self.train_epoch = 40
        self.dm = 0
        self.worker_count = 7


        labeled_sentences = []
        df_tags = []

        if isinstance(Y, basestring):
            df_tags.append(Y)
        if isinstance(Y, list):
            df_tags = Y
        elif not isinstance(Y, list):
            raise TypeError
        self.df = df
#         print(self.df)
        self.x = X
        self.y = Y
        self.df_tags = df_tags
        self.testseries = df[df_tags[0]].unique()
        self.testseries_name = df_tags[0]

    def score(self, tokenize_f, verbose=False):

        df = self.df
        X = self.x
        Y =self.y
        self.verbose = verbose
        if 'basestring' not in globals():
            basestring = str

        labeled_sentences = []
        df_tags = []

        if isinstance(Y, basestring):
            df_tags.append(Y)
        if isinstance(Y, list):
            df_tags = Y
        elif not isinstance(Y, list):
            raise TypeError



        if verbose:
            print("splitting train and test")
        train, test = train_test_split(self.df, shuffle = True)

        if verbose:
            print("labeling sentences")
        for index, datapoint in train.iterrows():
            tokenized_words = tokenize_f(datapoint[X].lower())
            labeled_sentences.append(TaggedDocument(words=tokenized_words, tags=[datapoint[i] for i in df_tags]))

        model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vector_size,
                                              window_size=self.window_size,
                                              min_count=self.min_count,
                                              sampling_threshold=self.sampling_threshold,
                                              negative_size=self.negative_size,
                                              train_epoch=self.train_epoch,
                                              dm=self.dm,
                                              worker_count=self.worker_count)
        if verbose:
            print("training model")
        model.build_vocab(labeled_sentences)
        model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model

        if verbose:
            print("making predictions")
        test_results = self.predict(test[X])
        if verbose:
            print("Scoring results")
        print("Label Score: ")

        '''
        ct = collections.Counter()
        cor = collections.Counter()

        for real, inferred in zip(test[self.testseries_name], test_results):
            ct[real] += 1

            if real == inferred:
                cor[inferred] += 1

        for i in ct:
            print("%s: %f %%" % (i, cor[i] / ct[i]))
        '''

        f1 = f1_score(test[self.testseries_name], test_results, average=None)
        print(f1)  # Uses train test split to get score
        print("Accuracy Score: ")
        acc = accuracy_score(test[self.testseries_name], test_results)
        print(acc)        # Uses train test split to get score

        return Score(
            overall = acc,
            positive = f1[2],
            negative = f1[0],
            neutral = f1[1]
        )

    def predict_text_main(self, document, col=None):  # takes in a string and infers vector and returns vectors and distance
        if col == None:
            col = self.df_tags[0]
        tokenized_words = re.findall(self.w, document.lower())
        inferred_vector = TaggedDocument(words=tokenized_words, tags=["inferred_vector"])[0]
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
#         print([rec for rec in sims if rec[0] in set(self.df[self.df_tags[0]].unique())])
        return [rec for rec in sims if rec[0] in set(self.df[col].unique())][0][0]

    def predict(self, X):  # Takes a series of text and returns a series of predictions
        if self.verbose:
            from tqdm import tqdm
            tqdm.pandas()
            return X.progress_apply(self.predict_text_main)
        else:
            return X.apply(self.predict_text_main)

'''

import contextlib
import itertools
import math
import statistics
import pickle

from gensim.models.doc2vec import Doc2Vec
import pymongo
from sklearn.model_selection import train_test_split

from cleantext import clean_data

tags = {"positive", "negative", "neutral", "irrelevant"}

def test(training_data):
    training_data, test_data = train_test_split(training_data)

    d2v = Doc2Vec(
        # Params from https://arxiv.org/pdf/1607.05368.pdf
    #    vector_size = 300,
    #    window_size = 15,
    #    min_count = 2,
    #    sampling_threshold = 1e-4,
    #    negative_size = 5,
    #    train_epoch = 50,
    #    dm = 0,
    #    worker_count = 7
    )

    d2v.build_vocab(datum.to_labeledsentence() for datum in training_data)
    d2v.train((datum.to_labeledsentence() for datum in training_data), total_examples = d2v.corpus_count, epochs = 5)

    loc = {k: math.sqrt(math.fsum(d2v.docvecs[k] ** 2)) for k in tags}
    correct = 0

    for datum in test_data:
        vec = d2v.infer_vector(datum.tokens)

        #predicted_loc = math.sqrt(math.fsum(vec ** 2))
        #predicted = max(tags, key = lambda k: math.fsum(vec * d2v.docvecs[k]) / (predicted_loc * loc[k]))

        possibilities = d2v.docvecs.most_similar([vec], topn = len(d2v.docvecs))
        predicted = max(possibilities, key = lambda x: x[1])[0]

        #print(datum.text)
        #print("Inferred:", predicted)
        #print("Actual:  ", datum.tags[0])

        if predicted in datum.tags:
            correct += 1

    #print("%f %% correct" % (100 * (correct / len(test_data))))
    return 100 * (correct / len(test_data))

if __name__ == "__main__":
    with contextlib.closing(pymongo.MongoClient()) as conn:
        collnames = ["LabeledStatuses_MiscTechCompanies_C", "LabeledStatuses_Sandy_K"]
        db = conn["twitter"]

        results = [r for r in db[collname].find() for collname in collnames]

        with open("training_data.pkl", "wb") as fd:
            pickle.dump(results, fd)
        exit()

        training_data, test_data = clean_data([], [])


    # Ignore test data for now
    test_data.close()

    assert all(any(x in tags for x in datum.tags) for datum in training_data)

    print("%f %% correct" % statistics.mean(test(training_data) for _ in range(15)))
'''
import contextlib

import pandas as pd
import pymongo
import nltk.corpus
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

stopwords = set(nltk.corpus.stopwords.words("english"))
stemmer = SnowballStemmer("english")

with contextlib.closing(pymongo.MongoClient()) as conn:
    coll = conn["twitter"]["LabeledStatuses_MiscTechCompanies_C"]
    records = [r["original"] for r in coll.find(projection = ["original"]) if r["original"]["TweetText"] is not None and r["original"]["Sentiment"] != "irrelevant"]

print("Sample size:", len(records))
df = pd.DataFrame(records)

def score_iter(tokenize_f,  n):
    ret = Score(0, 0, 0, 0)
    for _ in range(n):
        ret += doc2vec(df, "TweetText", ["Sentiment", "Topic"]).score(tokenize_f, verbose = True)
    ret = Score(
        overall = ret.overall / n,
        positive = ret.positive / n,
        negative = ret.negative / n,
        neutral = ret.neutral / n,
    )
    return ret

N = 1

def print_score(score, desc):
    print(desc + ":")
    print("\tOverall:  ", score.overall, "%")
    print("\tPositive: ", score.positive, "%")
    print("\tNegative: ", score.negative, "%")
    print("\tNeutral:  ", score.neutral, "%")
    print("")

regex_tokenizer = re.compile("\w+", re.I)
nltk_tokenizer = TweetTokenizer(preserve_case = False)

regex_identity_score = score_iter(regex_tokenizer.findall,  N)
regex_stem_score = score_iter(lambda x: [stemmer.stem(w) for w in regex_tokenizer.findall(x)],  N)
regex_rmstop_score = score_iter(lambda x: [w for w in regex_tokenizer.findall(x) if w not in stopwords],  N)
regex_stem_and_rmstop_score = score_iter(lambda x: [stemmer.stem(w) for w in regex_tokenizer.findall(x) if w not in stopwords],  N)

nltk_identity_score = score_iter(nltk_tokenizer.tokenize,  N)
nltk_stem_score = score_iter(lambda x: [stemmer.stem(w) for w in nltk_tokenizer.tokenize(x)],  N)
nltk_rmstop_score = score_iter(lambda x: [w for w in nltk_tokenizer.tokenize(x) if w not in stopwords],  N)
nltk_stem_and_rmstop_score = score_iter(lambda x: [stemmer.stem(w) for w in nltk_tokenizer.tokenize(x) if w not in stopwords],  N)

#def print_results(score, desc):
#    print(desc + ":")
#    print("\tOverall: %f %%" % )

print("")
print("-------")
print("RESULTS")
print("-------")
print("")
print_score(regex_identity_score, "Using regexes, without any modifications")
print_score(regex_stem_score, "Using regexes, with stemming")
print_score(regex_rmstop_score, "Using regexes, with removal of stop words")
print_score(regex_stem_and_rmstop_score, "Using regexes, with stemming and removal of stop words")
print_score(nltk_identity_score, "Using NLTK, without any modifications")
print_score(nltk_stem_score, "Using NLTK, with stemming")
print_score(nltk_rmstop_score, "Using NLTK, with removal of stop words")
print_score(nltk_stem_and_rmstop_score, "Using NLTK, with stemming and removal of stop words")
'''
print("Using NLTK:")
print("\tWithout any modifications:              ", nltk_identity_score, "%")
print("\tWith stemming:                          ", nltk_stem_score, "%")
print("\tWith removal of stop words:             ", nltk_rmstop_score, "%")
print("\tWith stemming and removal of stop words:", nltk_stem_and_rmstop_score, "%")
'''

