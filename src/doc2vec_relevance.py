import re
import gensim
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MyClass:

    def __init__(self):
        self.w = re.compile("\w+", re.I)

        #Loading Data
        sandy = pd.read_csv(
            "/home/manny/PycharmProjects/TweetAnalysis/relevance_labels/2012_Sandy_Hurricane-ontopic_offtopic.csv")
        sandy["Category"] = "sandy"
        # sandy = sandy.drop(sandy.query('Label == 0').sample(frac=.00).index)
        man1 = pd.read_csv("/home/manny/PycharmProjects/TweetAnalysis/relevance_labels/manually_labelled_data.csv")
        man2 = pd.read_csv(
            "/home/manny/PycharmProjects/TweetAnalysis/relevance_labels/Manually_labelled_data_2.csv")
        manual_train = pd.concat([man1, man2], sort=False, ignore_index=True)
        manual_train = manual_train.dropna()
        manual_train["Category"] = "curent"
        manual_train = pd.concat([manual_train, sandy], sort=False, ignore_index=True)
        self.manual_train = manual_train
        merged = self.label_sentences(manual_train)
        self.merged = merged
        #Combines data, drops NA


        #Trains Model
        model = gensim.models.doc2vec.Doc2Vec(vector_size=300,
                                              window_size=15,
                                              min_count=2,
                                              sampling_threshold=1e-5,
                                              negative_size=5,
                                              train_epoch=40,
                                              dm=0,
                                              worker_count=1)
        model.build_vocab(merged)
        model.train(merged, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model

    def score(self):
        #Uses train test split to get score
        train, test = train_test_split(self.merged, shuffle=True)

        model = gensim.models.doc2vec.Doc2Vec(vector_size=300,
                                              window_size=15,
                                              min_count=2,
                                              sampling_threshold=1e-5,
                                              negative_size=5,
                                              train_epoch=40,
                                              dm=0,
                                              worker_count=1)
        model.build_vocab(train)
        model.train(train, total_examples=model.corpus_count, epochs=model.epochs)

        total = 0
        correct = 0
        for line in test:
            if line[1][0] == self.predict_taggedtext(line[0]):
                correct = correct + 1
            total = total + 1

        print("Mean Accuracy", (correct / total))

    def predict_taggedtext(self, document):  #takes in a taged document and infers vector and returns whether it is releveant or not (1 or 0)
        # tokenized_words = re.findall(w, document.lower())
        # inferred_vector = TaggedDocument(words=tokenized_words, tags=["Vector"])[0]
        inferred_vector = document
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        confidence_dict = {}
        for pair in sims:
            confidence_dict[pair[0]] = pair[1]
        if confidence_dict[1] >= confidence_dict[0]:
            return 1
        else:
            return 0

    def predict_text(self, document):  #takes in a string and infers vector and returns whether it is releveant or not (1 or 0)
        tokenized_words = re.findall(self.w, document.lower())
        inferred_vector = TaggedDocument(words=tokenized_words, tags=["Vector"])[0]
        inferred_vector = document
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        confidence_dict = {}
        for pair in sims:
            confidence_dict[pair[0]] = pair[1]
        if confidence_dict[1] >= confidence_dict[0]:
            return 1
        else:
            return 0

    def label_sentences(self, df): #Takes a pandas dataframe, loops and makes a list of sentences that are split by word and adds tags of category and label
        labeled_sentences = []
        for index, datapoint in df.iterrows():
            tokenized_words = re.findall(self.w, datapoint["Tweet"].lower())
            labeled_sentences.append(
                TaggedDocument(words=tokenized_words, tags=[int(datapoint["Label"]), str(datapoint["Category"])]))
        return labeled_sentences

    def predict(self, X): #Takes a series of text and returns a series of predictions
        return X.apply(self.predict_text)




x = MyClass()

#My score
x.score()

man1 = pd.read_csv("/home/manny/PycharmProjects/TweetAnalysis/relevance_labels/manually_labelled_data.csv")
man1 = man1.dropna()

man1["predictions"] = x.predict(man1["Tweet"])

print(accuracy_score(man1["Label"], man1["predictions"]))
