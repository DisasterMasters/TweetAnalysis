
import csv
import os

import pandas as pd
from keras.layers import Dense, Activation, Dropout, Embedding, Bidirectional
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

training_data_path = r"/home/manny/PycharmProjects/TweetAnalysis/florida/training_data/"

tweets = []
labels_list = []

#reading in training data
for dirs, subdirs, files in os.walk(training_data_path):  #all data for supervised learning should be put in this directory
    for fname in files:
        file = open(dirs + "/" + fname, "r")
        # print(fname)
        csv_read = csv.reader(file)
        header = csv_read.next()
        for row in csv_read:
            tweet = row[0]
            tweets.append(tweet)


train = pd.read_csv('/home/manny/PycharmProjects/TweetAnalysis/DATA/T-06-Hurricane_Sandy_Labeled/2012_Sandy_Hurricane-ontopic_offtopic.csv', names = ['tweet id', 'tweet' , 'label'])




train['label'] = train['label'].map({'on-topic': int(1), 'off-topic': int(0)})
train['label'] = train['label'].fillna(value=0)
print(train.label.unique())

y = train['label']
X = train['tweet']

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')

tfidf = tfidf.fit(tweets)
tfidf = tfidf.fit(X)


# X_vectorized = tfidf.transform(X)


sum = 0
# for i in range(5):
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
#
#     X_train_word_features = tfidf.transform(X_train)
#
#     # transform the test features to sparse matrix
#     test_features = tfidf.transform(X_test)
#
#     print(X_train_word_features.shape)
#     print(y_train.shape)
#     print(test_features.shape)
#     print(y_test.shape)
#
#     # input("K")
#     model = RandomForestClassifier()
#     model.fit(X_train_word_features, y_train)
#     score = model.score(test_features, y_test)
#     sum += score
#     print("test score: " + str(score))
#
# print("=================" + "\n" +  "Average Score: " + str(sum /5))


X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train_word_features = tfidf.transform(X_train)

# transform the test features to sparse matrix
test_features = tfidf.transform(X_test)

print(X_train_word_features.shape)
print(y_train.shape)
print(test_features.shape)
print(y_test.shape)


# batch_size = 32
# epochs = 3
#
#
# # Build the model
# model = Sequential()
# # .add(Embedding(3383, 140))
#
# model.add(Dense(512, input_shape=(3383,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('softmax'))
#
# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
# history = model.fit(X_train_word_features, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_split=0.1)
#
#
# # Evaluate the accuracy of our trained model
# score = model.evaluate(test_features, y_test,
#                        batch_size=batch_size, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
#

# model = Sequential()
# model.add(Embedding(3383, output_dim=256))
# model.add(Bidirectional(LSTM(64)))
# model.add(Dropout(.5))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer="adam",
#               metrics=['accuracy'])
#
# history = model.fit(X_train_word_features, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_split=0.1)
#
#
# # Evaluate the accuracy of our trained model
# score = model.evaluate(test_features, y_test,
#                        batch_size=batch_size, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


## create model
model_glove = Sequential()
model_glove.add(Embedding(3383, 140))
model_glove.add(Dropout(0.2))
# model_glove.add(Conv1D(64, 5, activation='relu'))
# model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## Fit train data
model_glove.fit(X_train_word_features, y_train, validation_split=0.4, epochs=3, verbose=1)

# Evaluate the accuracy of our trained model
score = model_glove.evaluate(test_features, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])