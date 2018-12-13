import re
import gensim
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


class doc2vec:

    def __init__(self, df, X, Y):
        self.w = re.compile("\w+", re.I)
        if 'basestring' not in globals():
            basestring = str

        # Hyperparameters : https://arxiv.org/pdf/1607.05368.pdf
        self.vector_size = 200
        self.window_size = 15
        self.min_count = 2
        self.sampling_threshold = 1e-5
        self.negative_size = 5
        self.train_epoch = 40
        self.dm = 0
        self.worker_count = 1


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
        self.testseries = df[df_tags[0]].unique()
        self.testseries_name = df_tags[0]

        for index, datapoint in df.iterrows():
            tokenized_words = re.findall(self.w, datapoint[X].lower())
            labeled_sentences.append(TaggedDocument(words=tokenized_words, tags=[datapoint[i] for i in df_tags]))
        model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vector_size,
                                              window_size=self.window_size,
                                              min_count=self.min_count,
                                              sampling_threshold=self.sampling_threshold,
                                              negative_size=self.negative_size,
                                              train_epoch=self.train_epoch,
                                              dm=self.dm,
                                              worker_count=self.worker_count)
        model.build_vocab(labeled_sentences)
        model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model

    def score(self):

        df = self.df
        X = self.x
        Y =self.y
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



        train, test = train_test_split(self.df, shuffle=True)


        for index, datapoint in train.iterrows():
            tokenized_words = re.findall(self.w, datapoint[X].lower())
            labeled_sentences.append(TaggedDocument(words=tokenized_words, tags=[datapoint[i] for i in df_tags]))

        model = gensim.models.doc2vec.Doc2Vec(vector_size=self.vector_size,
                                              window_size=self.window_size,
                                              min_count=self.min_count,
                                              sampling_threshold=self.sampling_threshold,
                                              negative_size=self.negative_size,
                                              train_epoch=self.train_epoch,
                                              dm=self.dm,
                                              worker_count=self.worker_count)
        model.build_vocab(labeled_sentences)
        model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model

        test['results'] = self.predict(test[X])
        print(f1_score(test[self.testseries_name], test['results'], average=None))        # Uses train test split to get score


    def predict_taggedtext(self,
                           document):  # takes in a taged document and infers vector and returns whether it is releveant or not (1 or 0)
        inferred_vector = document
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        return sims

    def predict_text(self, document):  # takes in a string and infers vector and returns vectors and distance
        tokenized_words = re.findall(self.w, document.lower())
        inferred_vector = TaggedDocument(words=tokenized_words, tags=["inferred_vector"])[0]
        inferred_vector = document
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        return sims

    def predict_text_main(self, document):  # takes in a string and infers vector and returns vectors and distance
        tokenized_words = re.findall(self.w, document.lower())
        inferred_vector = TaggedDocument(words=tokenized_words, tags=["inferred_vector"])[0]
        inferred_vector = document
        inferred_vector = self.model.infer_vector(inferred_vector)
        sims = self.model.docvecs.most_similar([inferred_vector], topn=len(self.model.docvecs))
        dct = dict(sims)

        min = [self.testseries[0], dct.get(self.testseries[0])]
        for lables in self.testseries:
            if dct.get(lables) > min[1]:
                min[0] = lables
                min[1] = dct.get(lables)
        return min[0]

    def label_sentences(self, df, X, Y):
        # trick for py2/3 compatibility
        if 'basestring' not in globals():
            basestring = stzfr

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
        return X.apply(self.predict_text_main)


