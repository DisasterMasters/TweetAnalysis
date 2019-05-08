import collections
import contextlib
import math
import re
import pickle
import statistics
from email.utils import format_datetime
import csv
import sys

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
            tokenizer = TweetTokenizer(preserve_case = False)
            stopwords = frozenset(nltk.corpus.stopwords.words("english"))
            stemmer = SnowballStemmer("english")

            tokenize_f = lambda s: [stemmer.stem(w) for w in tokenizer.tokenize(s) if w not in stopwords]

            #tokenize_f = re.compile("\w+").findall

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

    with opentunnel(), opendb() as db:
        labeled_text = []
        labeled_tags = []

        with opencoll(db, "LabeledStatuses_Power_A") as coll:
            for r in coll.find(projection = ["text", "extended_tweet.full_text", "retweeted_status.extended_tweet.full_text", "tags"]):
                labeled_text.append(getcleantext(r))
                labeled_tags.append(r["tags"])

        model = d2v(labeled_text, labeled_tags, tags_to_test = [{"relevant", "irrelevant"}])

        with opencoll(db, "Statuses_MiscPower_A", cleanup = False) as coll, PickleMarshaller(sys.argv[1]) as marshaller:
            ids = set()

            while True:
                try:
                    for r in coll.aggregate([{"$sample": {"size": 1000}}]):
                    #for r in coll.find({"created_at": {"$gte": datetime.datetime(2019, 3, 14), "$lt": datetime.datetime(2019, 3, 15)}}):
                        id = r["retweeted_status"]["id"] if "retweeted_status" in r else r["id"]

                        if id in ids:
                            continue

                        ids.add(id)

                        text = getcleantext(r)
                        r["tags"] = model.infer(text)

                        if "relevant" not in r["tags"]:
                            continue

                        print("\n" + getnicetext(r))

                        relevant = input("\nIs this relevant? (yes/no/\033[4ms\033[0mkip) ").lower().strip()

                        if relevant in {"yes", "y", "true", "t", "on"}:
                            r["tags"] = ["relevant"]
                            marshaller.add(r)
                        elif relevant in {"no", "n", "false", "f", "off"}:
                            r["tags"] = ["irrelevant"]
                            marshaller.add(r)

                except KeyboardInterrupt:
                    break
