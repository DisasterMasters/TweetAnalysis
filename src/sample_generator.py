import pandas as pd
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

common = corpus.merge(train, ignore_index=True)

new = corpus[(~corpus.Tweet.isin(common.Tweet)) & (~corpus.Tweet.isin(common.Tweet))]
x = tfidf.transform(new['Tweet'])
print(x.shape)
rfresults = clf.predict(x)

rfresults = pd.Series(rfresults, name="RF")
results = pd.concat([new, rfresults], axis=1, ignore_index=True)
print(results.head())

# results = results[results.Tweet != 0]
results.sample(1000).to_csv("Sample2.csv")
