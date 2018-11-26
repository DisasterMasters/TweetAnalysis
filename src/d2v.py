import csv
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import math

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
with open('applecorpus.csv',encoding="utf8") as csv_file:
   csv_reader = csv.reader(csv_file, quotechar='"', delimiter=',')
   rowNum = 0
   serror_count = 0
   rerror_count = 0
   for row in csv_reader:
      senti = 42
      if(len(row) != 2):
         rerror_count += 1
      if(row[0]=="positive"):
         senti = 1
      if(row[0]=="negative"):
         senti = -1
      if(row[0]=="neutral"):
         senti = 0
      if(senti == 42):
         serror_count += 1
      all_tweets.append(clean(row[1]))
      all_sent.append(str(senti))

#some error checking on file read in
if(serror_count != 0):
   print(str(serror_count)+" invalid sentiments read")
if(rerror_count != 0):
   print(str(rerror_count)+" invalid row reads")

#split data into testing and training, randomly shuffled
t_train, t_test, s_train, s_test = train_test_split(all_tweets, all_sent)

#label tweets with sentiment
def label(tweets,sentiment):
   labeled = []
   for i in range(len(tweets)):
      labeled.append(LabeledSentence(tweets[i],sentiment[i]))
   return labeled


#trying out doc2vec
d2v_model= Doc2Vec(alpha=.025, min_alpha=.025)
d2v_model.build_vocab(label(t_train+t_test, s_train+s_test))
d2v_model.train(label(t_train, s_train),total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
d2v_model.save('./syddidthis.d2v')
print(d2v_model.docvecs[-1])
vtest = []
for tweet in t_test:
   vtest.append(d2v_model.infer_vector(tweet))

print(d2v_model.docvecs)
neg_loc = math.sqrt(sum(d2v_model.docvecs[-1]*d2v_model.docvecs[-1]))
neu_loc = math.sqrt(sum(d2v_model.docvecs[0]*d2v_model.docvecs[0]))
pos_loc = math.sqrt(sum(d2v_model.docvecs[1]*d2v_model.docvecs[1]))

pred_sent=[]
for i in range(len(t_test)):
    pred_loc = math.sqrt(sum(vtest[i]*vtest[i]))
    is_neg = sum(vtest[i]*d2v_model.docvecs[-1])/pred_loc/neg_loc
    is_neu = sum(vtest[i]*d2v_model.docvecs[0])/pred_loc/neu_loc
    is_pos = sum(vtest[i] * d2v_model.docvecs[1])/pred_loc/pos_loc
    if(is_neg>is_neu and is_neg>is_pos):
        pred_sent.append(-1)
    elif(is_neu>=is_neg and is_neu>=is_pos):
        pred_sent.append(0)
    elif(is_pos>is_neu and is_pos>is_neg):
        pred_sent.append(1)

correct = 0
for i in range(0, len(pred_sent)):
   if(int(pred_sent[i]) == int(s_test[i])):
       correct += 1

print("ONE TIME RUN")
print(correct/len(pred_sent))