import glob
from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.models import Model


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
    model_name = r"/home/manny/PycharmProjects/TweetAnalysis/src/models/50features_1minwords_10context"
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        # print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        # print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in sentence_matrix]]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        # print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
    np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    return embedding_weights



def pad_sentences(sentence, padding_word=" "):
    sequence_length = 924
    num_padding = sequence_length - len(sentence)
    new_sentence = str(sentence) + padding_word * int((num_padding/6))
    while(len(new_sentence) < 924):
        new_sentence = new_sentence + " "
    return new_sentence


def build_input_data(sentence, vocabulary):
    """
    Maps sentence and labels to vectors based on a vocabulary.
    """
    for word in sentence:
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary) +1

    x = np.array([[vocabulary[word] for word in sentence]])
    return [x]


def load_data(text):

    vocabulary = {'/': 0, 'A': 1, 'D': 2, 'P': 3, '<': 4, '>': 5, ' ': 6, 'e': 7, 'a': 8, 't': 9, 'o': 10, 'n': 11, 'r': 12, 'i': 13, 's': 14, 'h': 15, 'l': 16, 'u': 17, 'd': 18, 'c': 19, 'y': 20, 'm': 21, '.': 22, 'g': 23, 'p': 24, 'w': 25, 'f': 26, 'T': 27, 'b': 28, 'S': 29, 'k': 30, 'I': 31, 'R': 32, ':': 33, '@': 34, 'H': 35, 'N': 36, 'v': 37, 'E': 38, "'": 39, 'O': 40, 'C': 41, '#': 42, 'M': 43, 'Y': 44, '!': 45, 'L': 46, 'W': 47, 'B': 48, ',': 49, '?': 50, 'G': 51, 'U': 52, 'F': 53, 'j': 54, 'x': 55, '_': 56, 'K': 57, 'J': 58, '1': 59, 'z': 60, '0': 61, '2': 62, 'V': 63, '-': 64, '3': 65, '5': 66, '&': 67, '4': 68, ';': 69, ')': 70, 'q': 71, '7': 72, '9': 73, '(': 74, '8': 75, '6': 76, 'Z': 77, 'X': 78, 'Q': 79, '\\': 80, '“': 81, '”': 82, '*': 83, '$': 84, '’': 85, '[': 86, ']': 87, '|': 88, '%': 89, '=': 90, '^': 91, '~': 92, '+': 93, '…': 94, '‘': 95, '\xa0': 96, '–': 97, '—': 98, '`': 99, 'é': 100, '¤': 101, '»': 102, '}': 103, '°': 104, '¢': 105, '•': 106, '®': 107, 'í': 108, '«': 109, 'ö': 110}
    sentence_padded = pad_sentences(text)
    # print(sentence_padded)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x = build_input_data(sentence_padded, vocabulary)

    # print(x)
    x = x[0]
    return  x, vocabulary,  text




def model_predict(model_type, text):


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
    num_epochs = 30

    # Prepossessing parameters
    sequence_length = 924
    max_words = 5000

    # Word2Vec parameters (see train_word2vec)
    min_word_count = 1
    context = 10

    #
    # ---------------------- Parameters end -----------------------


    # Data Preparation
    # print("Load data...")
    data, vocabulary_inv, x = load_data(text)

    # data.reshape(1, 924)
    # print("data shape:", data.shape)


    # if sequence_length != data.shape:
    #     print("Adjusting sequence length for actual size")
    #     sequence_length = data.shape

    # print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

    # Prepare embedding layer weights and convert inputs for static model
    # print("Model type is", model_type)
    if model_type == "CNN-non-static":
        embedding_weights = train_word2vec(x, vocabulary_inv, num_features=embedding_dim,
                                           min_word_count=min_word_count, context=context)

    elif model_type == "CNN-rand":
        embedding_weights = None
    else:
        raise ValueError("Unknown model type")

    input_shape = (924,)

    model_input = Input(shape=input_shape)


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
        # print("Initializing embedding layer with word2vec weights, shape", weights.shape)
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
        latest_file = sorted(list_of_files, key=lambda x: float(os.path.basename(x).replace('weights-improvement-', '').replace('.hdf5', '')))


        # print(latest_file[0])

        model.load_weights(latest_file[0])

    else:
        print("No saved checkpoint found")
    # Train the model
    predict = model.predict(data)
    predict = predict.max()
    return round(predict)

# prediciton = model_predict("CNN-non-static" ,"Hurricane")
# print(prediciton)