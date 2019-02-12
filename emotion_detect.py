import pandas as pd
from pd_doc2vec import doc2vec
#read excel file into df, replace emotion NaNs with 0
df = pd.read_excel('emotions.xlsx')
df["Emotion"].fillna(0, inplace=True)

#test doc2vec for emotion detection
print("EMOTION/NO-EMOTION SCORE")
df=df.applymap(str)
x = doc2vec(df, 'Tweet',['Emotion'])
print(x.score()) #accuaracy ranged from 64-84

#simplify data frame to just contain tweets with emotion and their value
df2 = df
df2.drop(columns = ["Opinion", "Emotion", "Sentiment", "Sarcasm"], axis = 'columns', inplace=True)
df2.dropna(axis = "index", inplace = True)

#Give it to doc2vec as is, even with funky labels
print("EMOTION CLASS SCORE")
df2 = df.applymap(str)
y = doc2vec(df2,"Tweet",["Emotion class"])
print(y.score())