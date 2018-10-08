import pandas as pd
from collections import Counter
import numpy as np
import randomforrest_filter

tfidf, clf = randomforrest_filter.build_model(True)
# rfresults = clf.predict(tfidf.transform(corpus))



corpus = pd.read_csv('corpus.csv')


corpus = corpus.dropna()
# print(corpus['Tweet'])
x = tfidf.transform(corpus['Tweet'])
print(x.shape)
rfresults = clf.predict(x)


print(rfresults)
print(Counter(rfresults))


from crisislex import Crisislex

c_lex = Crisislex()
lex = c_lex.lex

lexresults = corpus['Tweet'].apply(lambda x: '1.0' if any((txt in x) for txt in lex) else '0.0')

print(Counter(lexresults))


print(type(corpus))
lexresults = pd.Series(lexresults, name="Lex")
rfresults = pd.Series(rfresults, name="RF")
results = pd.concat([corpus, rfresults, lexresults], axis=1)


results.sample(500).to_csv("Sample.csv")
