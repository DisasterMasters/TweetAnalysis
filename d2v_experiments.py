import collections
import contextlib
import re

import pandas as pd
import pymongo
import nltk.corpus
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

stopwords = set(nltk.corpus.stopwords.words("english"))
stemmer = SnowballStemmer("english")

def d2v_run(
    df,
    x,
    y,
    tokenize_f,
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

    if ys_to_test is None:
        ys_to_test = set(tag_coloring.keys())

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

    training_data_docs = [TaggedDocument(tokenize_f(datum[x]), datum[y]) for _, datum in training_data.iterrows()]

    model.build_vocab(training_data_docs)
    model.train(training_data_docs, total_examples = model.corpus_count, epochs = model.epochs)

    def infer(x):
        vec = model.infer_vector(tokenize_f(x))
        sims = model.docvecs.most_similar([vec], topn = len(model.docvecs))

        ret = []

        for dset in tag_dsets:
            a = [s[0] for s in sorted(sims, key = lambda x: x[1]) if s[0] in dset & ys_to_test]

            if len(a) > 0:
                ret.append(a[-1])

        print(ret)
        return ret

    def actual(y):
        return [s for s in y if s in ys_to_test]

    actual_tags = test_data[y].map(actual)
    inferred_tags = test_data[x].map(infer)

    inferred_tags = inferred_tags.where([bool(x) for x in actual_tags]).dropna()
    actual_tags = actual_tags.where([bool(x) for x in actual_tags]).dropna()

    print(actual_tags)

    mlb = MultiLabelBinarizer()
    inferred_mlb = mlb.fit_transform(inferred_tags)
    actual_mlb = mlb.fit_transform(actual_tags)

    f1 = f1_score(inferred_mlb, actual_mlb, average = None)
    acc = accuracy_score(inferred_mlb, actual_mlb)

    return acc, {k: v for k, v in zip(sorted(ys_to_test), f1)}

with contextlib.closing(pymongo.MongoClient()) as conn:
    coll = conn["twitter"]["LabeledStatuses_MiscTechCompanies_C"]
    records = [r for r in coll.find(projection = ["text", "tags"])]

df = pd.DataFrame(records)
#df["sentiment"] = df["tags"].map(lambda x: [v for v in x if v in {"positive", "negative", "neutral"}][0])
#df["topic"] = df["tags"].map(lambda x: [v for v in x if v not in {"positive", "negative", "neutral"}][0])

acc, f1 = d2v_run(df, "text", "tags", re.compile("\w+", re.I).findall, {"positive", "negative", "neutral"})
print(acc)
print(f1)
exit()

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

