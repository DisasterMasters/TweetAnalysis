

import csv
import glob
import os

import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Embedding
from keras.layers import LSTM
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


X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train_word_features = tfidf.transform(X_train)

# transform the test features to sparse matrix
test_features = tfidf.transform(X_test)

print(X_train_word_features.shape)
print(y_train.shape)
print(test_features.shape)
print(y_test.shape)


## create model
model = Sequential()
model.add(Embedding(3383, 140))
model.add(Dropout(0.2))
# model.add(Conv1D(64, 5, activation='relu'))
# model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#
#
# Check point saver, saves weights and model progress
#
#
filepath = "weights-improvement-{loss:.2f}.hdf5"
model_directory = r'/home/manny/PycharmProjects/TweetAnalysis/florida/results/model_checkpoint/'
check_pointer = ModelCheckpoint(
    filepath=model_directory + filepath, verbose=1,
    save_best_only=True)

# Loads latest Checkpoints
#
#
#


list_of_files = glob.glob(model_directory + '*')  # * means all if need specific format then *.csv

if len(list_of_files) >= 2:
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)

    model.load_weights(latest_file)
else:
    print("No saved checkpoint found: beginning training")
    
    

## Fit train data
model.fit(X_train_word_features, y_train, validation_split=0.4, epochs=3, verbose=1, callbacks=[check_pointer, TensorBoard(log_dir='/home/manny/PycharmProjects/TweetAnalysis/florida/results/model_checkpoint/tensorboard_logs')])

# Evaluate the accuracy of our trained model
score = model.evaluate(test_features, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])