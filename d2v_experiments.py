import collections
import contextlib
import re
import pickle

import nltk.corpus
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

stopwords = set(nltk.corpus.stopwords.words("english"))
stemmer = SnowballStemmer("english")

def d2v_run(
    df,
    x,
    y,
    tokenize_f,
    ys_to_train = None,
    ys_to_test = None,
    rmstop = False,
    stem = False,
    vector_size = 300,
    window_size = 15,
    min_count = 1,
    sampling_threshold = 1e-4,
    negative_size = 5,
    train_epoch = 40,
    dm = 0,
    worker_count = 7,
):
    # Determine which tags are mutually exclusive by performing a graph
    # coloring, where the tags are vertices and two tags being in the same
    # document are edges
    tag_coloring = {}

    for _, datum in df.iterrows():
        for tag in datum[y]:
            if tag not in tag_coloring:
                tag_coloring[tag] = min(set(range(len(df))) - {tag_coloring[x] for x in datum[y] if x in tag_coloring})

    # Form disjoint sets by color
    tag_dsets = [{k for k, v in tag_coloring.items() if v == i} for i in range(max(tag_coloring.values()) + 1)]

    if ys_to_train is None:
        ys_to_train = set(tag_coloring.keys())

    if ys_to_test is None:
        ys_to_test = set(tag_coloring.keys())

    ys_to_train = ys_to_train | ys_to_test

    training_data, test_data = train_test_split(df, shuffle = True)

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

    for _, datum in training_data.iterrows():
        # Only add data points whose tags overlap with ys_to_train to the
        # training data set
        tags = set(datum[y]) & ys_to_train

        if tags:
            training_data_docs.append(TaggedDocument(tokenize_f(datum[x]), list(tags)))

    model.build_vocab(training_data_docs)
    model.train(training_data_docs, total_examples = model.corpus_count, epochs = model.epochs)

    def infer(x):
        vec = model.infer_vector(tokenize_f(x))
        sims = model.docvecs.most_similar([vec], topn = len(model.docvecs))

        ret = []

        for dset in tag_dsets:
            # Append the highest-rated element for each disjoint set
            a = [s[0] for s in sorted(sims, key = lambda x: x[1]) if s[0] in dset & ys_to_test]

            if len(a) > 0:
                ret.append(a[-1])

        return ret

    def actual(y):
        return list(set(y) & ys_to_test)

    inferred_tags = test_data[x].map(infer)
    actual_tags = test_data[y].map(actual)

    mask = [bool(x) for x in actual_tags]
    inferred_tags = inferred_tags.where(mask).dropna()
    actual_tags = actual_tags.where(mask).dropna()

    mlb = MultiLabelBinarizer()
    inferred_mlb = mlb.fit_transform(inferred_tags)
    actual_mlb = mlb.fit_transform(actual_tags)

    f1 = f1_score(inferred_mlb, actual_mlb, average = None)
    acc = accuracy_score(inferred_mlb, actual_mlb)

    return acc, {k: v for k, v in zip(sorted(ys_to_test), f1)}

try:
    with open("labeled_statuses.pkl", "rb") as fd:
        df = pickle.load(fd)
except FileNotFoundError:
    import pymongo

    with contextlib.closing(pymongo.MongoClient()) as conn:
        coll = conn["twitter"]["LabeledStatuses_MiscTechCompanies_C"]
        records = [r for r in coll.find(projection = ["text", "tags"]) if "irrelevant" not in r["tags"]]

    df = pd.DataFrame(records)

    with open("labeled_statuses.pkl", "wb") as fd:
        pickle.dump(df, fd)

regex_tokenizer = re.compile("\w+", re.I)
nltk_tokenizer = TweetTokenizer(preserve_case = False)

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
    acc, f1 = d2v_run(df, "text", "tags", method, ys_to_test = {"positive", "negative", "neutral"})

    print(name + ":")
    print("\tOverall:  %f%%" % acc)
    print("\tPositive: %f%%" % f1["positive"])
    print("\tNegative: %f%%" % f1["negative"])
    print("\tNeutral:  %f%%" % f1["neutral"])
    print()

