import csv
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import math
import pandas as pd

#get rid of urls, update to get rid of garbage characters? deal with emojis?
def clean(tw):
   s = tw.split()
   s2 = ""
   for word in s:
      if "http://" not in word:
         s2+=(word+" ")
   return s2

#read in corpus
all_tweets = []
all_sent = []
serror_count = 0
c=0
# sanders analytics corpus
df = pd.read_csv('full-corpus.csv.gz', compression='gzip')
for index, row in df.iterrows():
    senti = 'forty-two'
    if (row[1] == "positive"):
        senti = 'POS'
        all_tweets.append(clean(row[4]))
        all_sent.append(senti)
    if (row[1] == "negative"):
        senti = 'NEG'
        all_tweets.append(clean(row[4]))
        all_sent.append(senti)
    if (row[1] == "neutral"):
        senti = 'NEU'
        all_tweets.append(clean(row[4]))
        all_sent.append(senti)
    if (senti == 'forty-two'):
        serror_count += 1

#Dr.Caragea's corpus
df2 = pd.read_csv('ClassifiedTweets.csv.gz', compression='gzip')
for index, row in df2.iterrows():
    senti = 'forty-two'
    if (row[6] == "positive"):
        senti = 'POS'
        all_tweets.append(clean(row[3]))
        all_sent.append(senti)
    if (row[6] == "negative"):
        senti = 'NEG'
        all_tweets.append(clean(row[3]))
        all_sent.append(senti)
    if(row[6] == "neutral"):
        senti = 'NEU'
        all_tweets.append(clean(row[3]))
        all_sent.append(senti)
    if (senti == 'forty-two'):
        serror_count += 1

#some error checking on file read in
if(serror_count != 0):
   print(str(serror_count)+" invalid sentiments read")

#split data into testing and training, randomly shuffled
t_train, t_test, s_train, s_test = train_test_split(all_tweets, all_sent)

#label tweets with sentiment
def label(tweets,sentiment):
   labeled = []
   for i in range(len(tweets)):
      labeled.append(LabeledSentence(tweets[i], [ sentiment[i] ]))

   return labeled

#trying out doc2vec
# for w in (3, 5, 10, 15, 20):
#     for ep in (5,10,15,20,30, 40, 50, 60):
d2v_model= Doc2Vec()
d2v_model.build_vocab(label(t_train, s_train))
d2v_model.train(label(t_train, s_train),total_examples=d2v_model.corpus_count, epochs=5)
vtest = []
for tweet in t_test:
    vtest.append(d2v_model.infer_vector(tweet))

neg_loc = math.sqrt(sum(d2v_model.docvecs['NEG']*d2v_model.docvecs['NEG']))
pos_loc = math.sqrt(sum(d2v_model.docvecs['POS']*d2v_model.docvecs['POS']))
neu_loc = math.sqrt(sum(d2v_model.docvecs['NEU']*d2v_model.docvecs['NEU']))

pred_sent=[]
for i in range(len(t_test)):
    pred_loc = math.sqrt(sum(vtest[i]*vtest[i]))
    is_neg = sum(vtest[i]*d2v_model.docvecs['NEG'])/pred_loc/neg_loc
    is_pos = sum(vtest[i]*d2v_model.docvecs['POS'])/pred_loc/pos_loc
    is_neu = sum(vtest[i]*d2v_model.docvecs['NEU'])/pred_loc/neu_loc
    if(is_neg>is_pos and is_neg>is_neu):
        pred_sent.append('NEG')
    elif(is_pos>is_neg and is_pos>is_neu):
        pred_sent.append('POS')
    else:
        pred_sent.append('NEU')

correct = 0
neg_cor = 0
pos_cor = 0
neu_cor = 0
neg_tt = 0
pos_tt = 0
neu_tt = 0

for i in range(0, len(pred_sent)):
    if (pred_sent[i] == s_test[i]):
        correct += 1
        if (pred_sent[i] == 'NEG'):
            neg_tt += 1
            neg_cor += 1
        elif (pred_sent[i] == 'POS'):
            pos_tt += 1
            pos_cor += 1
        elif (pred_sent[i] == 'NEU'):
            neu_tt += 1
            neu_cor += 1
    else:
        if (s_test[i] == 'NEG'):
            neg_tt += 1
        elif (s_test[i] == 'POS'):
            pos_tt += 1
        elif (s_test[i] == 'NEU'):
            neu_tt += 1
pos_acu = pos_cor / pos_tt
neg_acu = neg_cor / neg_tt
neu_acu = neu_cor / neu_tt

print(" acc= " + str(correct / len(pred_sent)))
print("Positive Accuracy = " + str(pos_acu) + " Negative Accuracy = " + str(neg_acu)+" Neutal Accuracy = "+str(neu_acu))
print("# Classified as Positive: " + str(pos_tt) + " # Classfied as Negative " + str(
    neg_tt) + " # Classifeid as Neutral " + str(neu_tt))

def pre(twes):
    pred= []
    for i in range(len(twes)):
        tws = d2v_model.infer_vector(twes[i])
        pred_loc = math.sqrt(sum(tws * tws))
        is_neg = sum(tws * d2v_model.docvecs['NEG']) / pred_loc / neg_loc
        is_pos = sum(tws * d2v_model.docvecs['POS']) / pred_loc / pos_loc
        is_neu = sum(tws * d2v_model.docvecs['NEU']) / pred_loc / neu_loc
        if (is_neg > is_pos and is_neg > is_neu):
            pred.append('negative')
        elif (is_pos > is_neg and is_pos > is_neu):
            pred.append('positive')
        else:
            pred.append('neutral')
    return pred

media = pd.read_excel("Local Media Tweets 1000 for sentiment analysis.xlsx")
media['Sentiment Prediction'] = 'unpredicted'
alltw = media['Tweet']
allsen = pre(media['Tweet'])
media['Sentiment Prediction'] = allsen
media.to_excel('media_sentiment.xlsx')