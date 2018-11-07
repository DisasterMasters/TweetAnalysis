import pandas as pd
from collections import Counter
import numpy as np
import randomforrest_filter
import glob

tfidf, clf = randomforrest_filter.build_model(True)
# rfresults = clf.predict(tfidf.transform(corpus))

data_source = r"/home/manny/PycharmProjects/TweetAnalysis/DATA/T-06-Hurricane_Sandy_Labeled/"

training_list = []
files = glob.glob(data_source + "/*.csv")
for fname in files:
    df = pd.read_csv(fname, index_col=None, header=0)
    training_list.append(df)

train = pd.concat(training_list, sort=False, ignore_index=True)

corpus = pd.read_csv('corpus.csv')


corpus = corpus.dropna()


common = corpus.merge(train)
new = corpus[(~corpus.Tweet.isin(common.Tweet))&(~corpus.Tweet.isin(common.Tweet))]
x = tfidf.transform(new['Tweet'])
print(x.shape)
rfresults = clf.predict(x)

rfresults = pd.Series(rfresults, name="RF")
results = pd.concat([new, rfresults], axis=1, ignore_index=True)
print(results.head())

common = corpus.merge(train)
print(common)
corpus = corpus[(~corpus.Tweet.isin(common.Tweet))&(~corpus.Tweet.isin(common.Tweet))]

# print(corpus['Tweet'])
x = tfidf.transform(corpus['Tweet'])
print(x.shape)
rfresults = clf.predict(x)


print(rfresults)
print(Counter(rfresults))


# from crisislex import Crisislex
#
# c_lex = Crisislex()
# lex = c_lex.lex
#
# # lexresults = corpus['Tweet'].apply(lambda x: '1.0' if any((txt in x) for txt in lex) else '0.0')
#
# print(Counter(lexresults))

#
# lexresults = pd.Series(lexresults, name="Lex")
rfresults = pd.Series(rfresults, name="RF")
results = pd.concat([corpus, rfresults], axis=1, ignore_index=True)


results.sample(1000).to_csv("/home/manny/PycharmProjects/TweetAnalysis/DATA/T-06-Hurricane_Sandy_Labeled/Manually_labelled_data_4.csv")
