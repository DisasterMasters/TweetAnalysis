"""
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

"""

import glob
import itertools
import os
from collections import Counter
from os.path import join, exists, split

import numpy as np
import pandas as pd
from gensim.models import word2vec
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split



def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.

    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {int: str}
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """
    model_dir = 'models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
    np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    return embedding_weights


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = str(sentence) + padding_word * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    return [x]


def load_data(data_source):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    # sentences, labels = load_data_and_labels()

    train = pd.read_csv(data_source, names=['tweet id', 'tweet', 'label'])

    train['label'] = train['label'].map({'on-topic': int(1), 'off-topic': int(0)})
    train['label'] = train['label'].fillna(value=0)
    y = train['label']
    sentences = train['tweet']

    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x = build_input_data(sentences_padded, vocabulary)
    x = x[0]
    vocabulary_inv_list = vocabulary_inv
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    x_nums = x[np.newaxis]
    x_nums = x_nums.transpose()

    x_nums = pd.DataFrame(x_nums[:, -1].tolist())
    x_nums = x_nums.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(x_nums, y)

    return X_train, X_test, y_train, y_test, vocabulary_inv, x



def main(model_type):
    np.random.seed(0)

    pathtoproject = r"/home/manny/PycharmProjects/TweetAnalysis/florida/results/nn_cleaning/"
    # model_type = "CNN-rand"  # CNN-rand|CNN-non-static

    # Data source
    data_source = r"/home/manny/PycharmProjects/TweetAnalysis/DATA/T-06-Hurricane_Sandy_Labeled/2012_Sandy_Hurricane-ontopic_offtopic.csv"

    # Model Hyperparameters
    embedding_dim = 50
    filter_sizes = (3, 8)
    num_filters = 10
    dropout_prob = (0.5, 0.8)
    hidden_dims = 50

    # Training parameters
    batch_size = 64
    num_epochs = 5

    # Prepossessing parameters
    sequence_length = 400
    max_words = 5000

    # Word2Vec parameters (see train_word2vec)
    min_word_count = 1
    context = 10

    #
    # ---------------------- Parameters end -----------------------


    # Data Preparation
    print("Load data...")
    x_train, x_test, y_train, y_test, vocabulary_inv, x = load_data(data_source)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    if sequence_length != x_test.shape[1]:
        print("Adjusting sequence length for actual size")
        sequence_length = x_test.shape[1]

    print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    # Prepare embedding layer weights and convert inputs for static model
    print("Model type is", model_type)
    if model_type == "CNN-non-static":
        embedding_weights = train_word2vec(x, vocabulary_inv, num_features=embedding_dim,
                                           min_word_count=min_word_count, context=context)

    elif model_type == "CNN-rand":
        embedding_weights = None
    else:
        raise ValueError("Unknown model type")

    # Build model
    input_shape = (sequence_length,)

    model_input = Input(shape=input_shape)

    # Static model does not have embedding layer
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

    z = Dropout(dropout_prob[0])(z)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(1, activation="sigmoid")(z)

    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Initialize weights with word2vec
    if model_type == "CNN-non-static":
        weights = np.array([v for v in embedding_weights.values()])
        print("Initializing embedding layer with word2vec weights, shape", weights.shape)
        embedding_layer = model.get_layer("embedding")
        embedding_layer.set_weights([weights])

    filepath = "weights-improvement-{loss:.2f}.hdf5"
    if model_type == "CNN-rand":
        model_directory = pathtoproject + 'cnn_rand_models/'
        tsboard_path = pathtoproject + 'tensorboard_logs/cnn_rand_30'

    else:
        model_directory = pathtoproject + 'cnn_non_static_models/'
        tsboard_path = pathtoproject + 'tensorboard_logs/cnn_non_static_30'
    check_pointer = ModelCheckpoint(
        filepath=model_directory + filepath, verbose=1,
        save_best_only=True)


    list_of_files = glob.glob(model_directory + '*')  # list of files and their path

    if len(list_of_files) >= 1:
        latest_file = sorted(list_of_files, key=lambda x: float(x.replace('/home/manny/PycharmProjects/TweetAnalysis/florida/results/nn_cleaning/cnn_rand_models/weights-improvement-', '').replace('.hdf5', '')))
        print(latest_file[0])

        model.load_weights(latest_file[0])
    else:
        print("No saved checkpoint found: beginning training")
    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
              validation_data=(x_test, y_test), verbose=2, callbacks=[check_pointer, TensorBoard(
            log_dir=tsboard_path)])

model_type = "CNN-rand"  # CNN-rand|CNN-non-static

main("CNN-rand")

# model_type = "CNN-non-static"  # CNN-rand|CNN-non-static

main("CNN-non-static")# model_type = "CNN-rand"  # CNN-rand|CNN-non-static

# main("CNN-rand")
#
# # model_type = "CNN-non-static"  # CNN-rand|CNN-non-static
#
# main("CNN-non-static")



