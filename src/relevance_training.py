from pymongo import MongoClient
from tqdm import tqdm
import pandas as pd
from pd_doc2vec import doc2vec

client = MongoClient()
client = MongoClient('da1.eecs.utk.edu', 27017) 
#Connects to the MongoDB, make sure youre SSH'ed into the docker

i =0
relevancedata = []

for post in client['twitter']['LabeledStatuses_MiscRelevant_C'].find({}):  
    #Extracts all the entries with a sentiment field
    #Updates each data with the name of the collection it comes from
    relevancedata.append(post) 

df = pd.DataFrame(relevancedata)
# pulls data into the Pandas DataFrame


print(df.head(3))
print("Setting vairables")

print(df.columns.values)

def remove_list(funnyboi):
    return funnyboi[0]

df.tags = df.tags.apply(remove_list)

print(df.tags.value_counts())

print(df.isna().sum())

df.text = df.text.astype(str)

# We pass the class 3 fields:
# 1. The DataFrame
# 2. The X value, the Text (Pandas Series)
# 3. The Y values, the labels that correspond to the text (Pandas Series)
#                   It can be a list of the names of the columns or one as a string

x = doc2vec(df, "text", ["tags"])

print("Scoring Model")

#returns scores of each labels accuracy from the first column passed into 
# the Y or 3 arguments, in this example it was "main"
# this uses sklearn.metrics.f1_score, you can pass in 
x.score(verbose=True)

tweets = pd.read_csv("tweets.csv")
print(tweets.head(3))

print(tweets.Relevance.value_counts())

print(tweets.isna().sum())

i = 0
for index, datapoint in tweets.iterrows():
    if datapoint["Relevance"] != 1:
        datapoint["Relevance"] =0
    i = i+1
    if i > 395:
        break
print(tweets.Relevance.value_counts())

tweets  =  tweets[["text", "Relevance"]]
print(tweets.isna().sum())

tweets = tweets.dropna()
print(tweets.isna().sum())

# We pass the class 3 fields:
# 1. The DataFrame
# 2. The X value, the Text (Pandas Series)
# 3. The Y values, the labels that correspond to the text (Pandas Series)
#                   It can be a list of the names of the columns or one as a string

x = doc2vec(tweets, "text", ["Relevance"])

print("Scoring Model")

#returns scores of each labels accuracy from the first column passed into 
# the Y or 3 arguments, in this example it was "main"
# this uses sklearn.metrics.f1_score, you can pass in 
x.score(verbose=True)

frames = []
df  =  df[["text", "tags"]]

frames.append(df)
frames.append(tweets)
result = pd.concat(frames)
print(result.columns.values)

result['Relevance'].fillna(result['tags'], inplace=True)
result  =  result[["text", "tags"]]
print(result.columns.values)

print(result.isna().sum())

print(result.tags.value_counts())

result['tags'].fillna(0.0, inplace=True)
print(result.isna().sum())

# We pass the class 3 fields:
# 1. The DataFrame
# 2. The X value, the Text (Pandas Series)
# 3. The Y values, the labels that correspond to the text (Pandas Series)
#                   It can be a list of the names of the columns or one as a string

x = doc2vec(result, "text", ["tags"])

print("Scoring Model")

#returns scores of each labels accuracy from the first column passed into 
# the Y or 3 arguments, in this example it was "main"
# this uses sklearn.metrics.f1_score, you can pass in 
x.score(verbose=True)

