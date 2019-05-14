import collections
import contextlib
import math
import re
import pickle
import statistics
from email.utils import format_datetime
import csv

import nltk.corpus
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
import pandas as pd
from numpy import nan as NaN

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from marshalling import PickleMarshaller
from common import *



def getcleantext(text):

    # Normalize Unicode
    cleantext = unicodedata.normalize("NFC", text)
    # Remove characters outside BMP (emojis)
    cleantext = "".join(c for c in cleantext if ord(c) <= 0xFFFF)
    # Remove newlines and tabs
    cleantext = cleantext.replace("\n", " ").replace("\t", " ")
    # Remove HTTP(S) link
    cleantext = re.sub(r"https?://\S+", "", cleantext)
    # Remove pic.twitter.com
    cleantext = re.sub(r"pic.twitter.com/\S+", "", cleantext)
    # Remove @handle at the start of the tweet
    cleantext = re.sub(r"\A(@[A-Za-z0-9_]{1,15} ?)*", "", cleantext)
    # Remove RT @handle:
    cleantext = re.sub(r"RT @[A-Za-z0-9_]{1,15}:", "", cleantext)
    # Strip whitespace
    cleantext = cleantext.strip()

    return cleantext

class d2v:
    def __init__(
        self,
        training_text,
        training_tags,
        tokenize_f = None,
        tags_to_train = None,
        tags_to_test = None,
        vector_size = 300,
        window_size = 15,
        min_count = 1,
        sampling_threshold = 1e-4,
        negative_size = 5,
        train_epoch = 40,
        dm = 0,
        worker_count = 7,
    ):
        training_data = list(zip(training_text, training_tags))

        if tokenize_f is None:
            #tokenizer = TweetTokenizer(preserve_case = False)
            #stopwords = frozenset(nltk.corpus.stopwords.words("english"))
            #stemmer = SnowballStemmer("english")

            tokenize_f = re.compile("\w+").findall

        if tags_to_train is None:
            tags_to_train = {tag for _, tags in training_data for tag in tags}

        if tags_to_test is None:
            tags_to_test = [tags_to_train]

        model = Doc2Vec(
            vector_size = vector_size,
            window_size = window_size,
            min_count = min_count,
            sampling_threshold = sampling_threshold,
            negative_size = negative_size,
            train_epoch = train_epoch,
            dm = dm,
            worker_count = worker_count
        )

        training_data_docs = []

        for text, tags in training_data:
            # Only add data points whose tags overlap with ys_to_train to the
            # training data set
            tags = list(set(tags) & tags_to_train)

            if tags:
                training_data_docs.append(TaggedDocument(tokenize_f(text), tags))

        model.build_vocab(training_data_docs)
        model.train(training_data_docs, total_examples = model.corpus_count, epochs = model.epochs)

        self.model = model
        self.tags = tags_to_test
        self.tokenize = tokenize_f

    def infer(self, text):
        vec = self.model.infer_vector(self.tokenize(text))
        sims = self.model.docvecs.most_similar([vec], topn = len(self.model.docvecs))

        ret = []

        for dset in self.tags:
            # Append the highest-rated element for each disjoint set
            a = [s for s in sims if s[0] in dset]

            if a:
                ret.append(max(a, key = lambda x: x[1])[0])

        return ret

    @staticmethod
    def test(df, x, y, *args, **kwargs):
        training_data, test_data = train_test_split(df, shuffle = True)

        model = d2v(training_data[x], training_data[y], *args, **kwargs)
        ys_to_test = set()

        for dset in model.tags:
            ys_to_test |= dset

        inferred_tags = test_data[x].map(model.infer)
        actual_tags = test_data[y].map(lambda y: list(set(y) & ys_to_test))

        mask = list(map(bool, actual_tags))
        inferred_tags = inferred_tags.where(mask).dropna()
        actual_tags = actual_tags.where(mask).dropna()

        mlb = MultiLabelBinarizer()
        inferred_mlb = mlb.fit_transform(inferred_tags)
        actual_mlb = mlb.fit_transform(actual_tags)

        f1 = f1_score(inferred_mlb, actual_mlb, average = None)
        acc = accuracy_score(inferred_mlb, actual_mlb)

        return acc, {k: v for k, v in zip(sorted(ys_to_test), f1)}

if __name__ == "__main__":

    stopwords = frozenset(nltk.corpus.stopwords.words("english"))
    stemmer = SnowballStemmer("english")

    regex_tokenizer = re.compile("\w+", re.I)
    nltk_tokenizer = TweetTokenizer(preserve_case = False)

    #regex = re.compile(r"RT @[A-Za-z0-9_]{1,15}: ")

    with opentunnel(), opendb() as db, opencoll(db, "LabeledStatuses_Hurricane_C") as coll:
        rs = list(coll.find(projection = ["text", "tags"]))

    df = pd.DataFrame(rs)
    df["text"] = df["text"].map(getcleantext)

    #text_series = df["extended_tweet"].map(f1, na_action = "ignore")
    #text_series.fillna(df["retweeted_status"].map(f2, na_action = "ignore"), inplace = True)
    #text_series.fillna(df["text"], inplace = True)
    #df["text"] = text_series
    #df.drop(columns = ["_id", "retweeted_status", "extended_tweet"], inplace = True)

    print(df)

    methods = {
        "Using whitespace splitting, without any modifications": lambda x: x.split(),
        "Using whitespace splitting, with stemming": lambda x: [stemmer.stem(w) for w in x.split()],
        "Using whitespace splitting, with removal of stop words": lambda x: [w for w in x.split() if w not in stopwords],
        "Using whitespace splitting, with stemming and removal of stop words": lambda x: [stemmer.stem(w) for w in x.split() if w not in stopwords],
        "Using regexes, without any modifications": regex_tokenizer.findall,
        "Using regexes, with stemming": lambda x: [stemmer.stem(w) for w in regex_tokenizer.findall(x)],
        "Using regexes, with removal of stop words": lambda x: [w for w in regex_tokenizer.findall(x) if w not in stopwords],
        "Using regexes, with stemming and removal of stop words": lambda x: [stemmer.stem(w) for w in regex_tokenizer.findall(x) if w not in stopwords],
        "Using NLTK, without any modifications": nltk_tokenizer.tokenize,
        "Using NLTK, with stemming": lambda x: [stemmer.stem(w) for w in nltk_tokenizer.tokenize(x)],
        "Using NLTK, with removal of stop words": lambda x: [w for w in nltk_tokenizer.tokenize(x) if w not in stopwords],
        "Using NLTK, with stemming and removal of stop words": lambda x: [stemmer.stem(w) for w in nltk_tokenizer.tokenize(x) if w not in stopwords]
    }

    for name, method in methods.items():
        overall = []
        relevant = []
        irrelevant = []

        for _ in range(20):
            acc, f1 = d2v.test(df, "text", "tags", tokenize_f = method, tags_to_test = [{"relevant", "irrelevant"}])

            overall.append(acc)
            relevant.append(f1["relevant"])
            irrelevant.append(f1["irrelevant"])

        print(name + ":")
        print("\tOverall:    %f%% (Variance %f)" % (statistics.mean(overall), statistics.variance(overall)))
        print("\tRelevant:   %f%% (Variance %f)" % (statistics.mean(relevant), statistics.variance(relevant)))
        print("\tIrrelevant: %f%% (Variance %f)" % (statistics.mean(irrelevant), statistics.variance(irrelevant)))
        print()



    '''
    with opentunnel(), opendb() as db:
        labeled_text = []
        labeled_tags = []

        with opencoll(db, "LabeledStatuses_Power_A") as coll:
            for r in coll.find(projection = ["text", "extended_tweet.full_text", "retweeted_status.extended_tweet.full_text", "tags"]):
                labeled_text.append(getcleantext(r))
                labeled_tags.append(r["tags"])

        model = d2v(labeled_text, labeled_tags, tags_to_test = [{"relevant", "irrelevant"}])

        with opencoll(db, "Statuses_MiscPower_A") as coll, PickleMarshaller("Relevant_Inferred.pkl") as marshaller:
            rt_ids = set()

            for r in coll.find(skip = 1000):
                if "retweeted_status" in r:
                    if r["retweeted_status"]["id"] in rt_ids:
                        continue

                    rt_ids.add(r["retweeted_status"]["id"])

                text = getcleantext(r)
                r["tags"] = model.infer(text)

                if "relevant" in r["tags"] and len(model.tokenize(text)) > 3:
                    print(getnicetext(r))
                    marshaller.add(r)
    '''
