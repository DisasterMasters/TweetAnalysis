# for i in range(0, 11, 1):
#     # print(i/10.0)
#     x = doc2vev_relevance()  # what
#
#     print(x.predict_text("Calling owners of short draft boats to rescue residents of Levittown "))
#     # My score
#     x.score()
#
#     man1 = x.manual_train
#     man1 = man1.dropna()
#
#     man1["predictions"] = x.predict(man1["Tweet"])
#
#     score = f1_score(man1["Label"], man1["predictions"], average=None)
#     print(score)
#     # print(score[0]+score[1])

import re
import gensim
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class doc2vec:

    def __init__(self, df, X, Y):
        self.w = re.compile("\w+", re.I)
        if 'basestring' not in globals():
            basestring = str

        labeled_sentences = []
        df_tags = []

        if isinstance(Y, basestring):
            df_tags.append(Y)
        if isinstance(Y, list):
            df_tags = Y
        elif not isinstance(Y, list):
            raise TypeError
        self.df = df
        self.x = X
        self.y = Y

        for index, datapoint in df.iterrows():
            tokenized_words = re.findall(self.w, datapoint[X].lower())
            labeled_sentences.append(TaggedDocument(words=tokenized_words, tags=[datapoint[i] for i in df_tags]))
        model = gensim.models.doc2vec.Doc2Vec(vector_size=200,
                                              window_size=15,
                                              min_count=2,
                                              sampling_threshold=1e-5,
                                              negative_size=5,
                                              train_epoch=40,
                                              dm=0,
                                              worker_count=1)
        model.build_vocab(labeled_sentences)
        model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model

    # def score(self):

    # # Uses train test split to get score
    # train, test = train_test_split(self.merged, shuffle=True)
    #
    # model = gensim.models.doc2vec.Doc2Vec(vector_size=200,
    #                                       window_size=15,
    #                                       min_count=2,
    #                                       sampling_threshold=1e-5,
    #                                       negative_size=5,
    #                                       train_epoch=40,
    #                                       dm=0,
    #                                       worker_count=1)
    # model.build_vocab(train)
    # model.train(train, total_examples=model.corpus_count, epochs=model.epochs)

    def predict_taggedtext(self,
                           document):  # takes in a taged document and infers vector and returns whether it is releveant or not (1 or 0)
        # tokenized_words = re.findall(w, document.lower())
        # inferred_vector = TaggedDocument(words=tokenized_words, tags=["Vector"])[0]
        inferred_vector = document
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        return sims
        # confidence_dict = {}
        # for pairaccuracy_score in sims:
        #     confidence_dict[pairaccuracy_score[0]] = pairaccuracy_score[1]
        # if confidence_dict[1] >= confidence_dict[0]:
        #     return 1
        # else:
        #     return 0

    def predict_text(self,
                     document):  # takes in a string and infers vector and returns whether it is releveant or not (1 or 0)
        tokenized_words = re.findall(self.w, document.lower())
        inferred_vector = TaggedDocument(words=tokenized_words, tags=["inferred_vector"])[0]
        inferred_vector = document
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        # confidence_dict = {}
        # for pair in sims:
        #     confidence_dict[pair[0]] = pair[1]
        # if confidence_dict[1] >= confidence_dict[0]:
        #     return 1
        # else:
        #     return 0
        return sims

    def label_sentences(self, df, X, Y):
        # trick for py2/3 compatibility
        if 'basestring' not in globals():
            basestring = str

        labeled_sentences = []
        df_tags = []

        if isinstance(Y, basestring):
            df_tags.append(Y)
        if isinstance(Y, list):
            df_tags = Y
        elif not isinstance(Y, list):
            raise TypeError
        self.df = df
        self.x = X
        self.y = Y

        for index, datapoint in df.iterrows():
            tokenized_words = re.findall(self.w, datapoint[X].lower())
            labeled_sentences.append(TaggedDocument(words=tokenized_words, tags=[datapoint[i] for i in df_tags]))
        return labeled_sentences

    def predict(self, X):  # Takes a series of text and returns a series of predictions
        return X.apply(self.predict_text)


# Loading Data
sandy = pd.read_csv(
    "/home/manny/PycharmProjects/TweetAnalysis/relevance_labels/2012_Sandy_Hurricane-ontopic_offtopic.csv")
sandy["Category"] = "Sandy"
sandy['Label'] = sandy['Label'].astype(str) + "_Sandy"
man1 = pd.read_csv("/home/manny/PycharmProjects/TweetAnalysis/relevance_labels/manually_labelled_data.csv")
man2 = pd.read_csv(
    "/home/manny/PycharmProjects/TweetAnalysis/relevance_labels/Manually_labelled_data_2.csv")
manual_train = pd.concat([man1, man2], sort=False, ignore_index=True)
manual_train = manual_train.dropna()
manual_train["Category"] = "Curent"
manual_train['Label'] = manual_train['Label'].astype(str) + "_Curent"
full_dataset = pd.concat([manual_train, sandy], sort=False, ignore_index=True)

x = doc2vec(full_dataset, "Tweet", ["Label", "Category"])
print(x.model)