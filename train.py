import os
import pickle

import numpy as np
from keras.callbacks import Callback
from keras.engine import Input, Model
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from progress.bar import Bar
from sklearn.metrics import f1_score, recall_score, precision_score

train_path = '/home/djstrong/projects/repos/cycloped-io/cycloped-io/statistical_classification/multi_5first_paragraphs_features_class.csv'
train_path = 'multi_5first_paragraphs_features_class_shuf.csv'
test_path = '/home/djstrong/projects/repos/cycloped-io/cycloped-io/statistical_classification/multi_5first_paragraphs_features_class_gold.csv'

from gensim import corpora

__author__ = 'djstrong'
import csv

GLOVE_DIR = '/home/djstrong/projects/repos/wiki-keras'
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 30


def load_embeddings(path):
    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    bar = Bar('Processing', suffix='%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=400000)
    for line in f:
        bar.next()
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    bar.finish()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def count_labels(path, labels_dict=None):
    if labels_dict is None:
        labels_dict = {}

    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            text = row[2]
            labels = row[-3].split(' ')
            for label in labels:
                if label not in labels_dict:
                    labels_dict[label] = len(labels_dict)
    return labels_dict


def text_generator(path):
    with open(path) as f:
        reader = csv.reader(f)
        bar = Bar('Processing', suffix='%(index)d/%(max)d %(percent).1f%% - %(eta_td)s', max=2799601)
        for row in reader:
            bar.next()
            text = row[2]
            yield text
        bar.finish()


def Xy_generator(generator):
    for batch_X, batch_y in generator:
        yield (batch_X, batch_y)


def pad_generator(generator, tokenizer, labels_dict, sequence_length=20):
    for batch_X, batch_y in generator:
        sequences = tokenizer.texts_to_sequences(batch_X)
        dataX = pad_sequences(sequences, maxlen=sequence_length)
        # print(len(batch_y))
        # print(len(batch_y[0]))
        # print(len(batch_y[0][0]))
        # print(len(batch_y[0][0][0]))
        # print(batch_y[0])
        # print(batch_y)
        dataY = to_categorical(np.asarray([[labels_dict[y] for y in sample] for sample in batch_y]), len(labels_dict))

        yield (dataX, dataY)


def batch_generator(generator, batch_size=32):
    batch_X = []
    batch_y = []
    for X, y in generator:
        batch_X.append(X)
        batch_y.append(y)
        if len(batch_X) == batch_size:
            yield (batch_X, batch_y)
            batch_X = []
            batch_y = []


def data_generator(path):
    while 1:
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[2]
                labels = row[-3].split(' ')
                yield (text, labels)

class LossHistory(Callback):
    def __init__(self, evaluator):
        self.evaluator=evaluator

    def on_epoch_end(self, epoch, logs={}):
        out = self.evaluator.evaluate(self.model)
        print(out)
        # logs['val_score'] = out[2]


class Evaluator:
    def __init__(self, test_data, labels_dict):
        self.test_data = test_data
        self.labels_dict=[v for k,v in sorted(labels_dict.items(), key=lambda e: e[1])]

    def evaluate(self, model, verbose=False):
        # print len(self.test_data), len(self.test_data[0])
        X_train, y_train = self.test_data
        predictions = model.predict(X_train, verbose=1)

        predictions[predictions<0.5]=0
        predictions[predictions>=0.5]=1
        labels = self.labels_dict

        # print(y_train)
        # print(predictions)
        print('Micro - ', 'Precision:', precision_score(y_train, predictions, average='micro', labels=labels), 'Recall:', recall_score(y_train, predictions, average='micro', labels=labels), 'F1:', f1_score(y_train, predictions, average='micro', labels=labels))
        print('Macro - ', 'Precision:', precision_score(y_train, predictions, average='macro', labels=labels), 'Recall:', recall_score(y_train, predictions, average='macro', labels=labels), 'F1:', f1_score(y_train, predictions, average='macro', labels=labels))
        return f1_score(y_train, predictions, average='macro', labels=labels)


try:
    tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
except IOError:
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text_generator(train_path))
    file = open('tokenizer.pickle', 'wb')
    pickle.dump(tokenizer, file)
    file.close()

print('Loaded tokenizer.')

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))

try:
    embedding_matrix = pickle.load(open('embedding_matrix.pickle', 'rb'))
except IOError:
    embeddings_index = load_embeddings(GLOVE_DIR)

    # prepare embedding matrix

    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    file = open('embedding_matrix.pickle', 'wb')
    pickle.dump(embedding_matrix, file)
    file.close()

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

try:
    labels_dict = pickle.load(open('labels_dict.pickle', 'rb'))
except IOError:
    labels_dict = count_labels(train_path)
    labels_dict = count_labels(train_path, labels_dict)
    file = open('labels_dict.pickle', 'wb')
    pickle.dump(labels_dict, file)
    file.close()

labels_dict = {"": 1, "Event": 2, "Organization": 3, "PhysicalDevice": 4, "Technology-Artifact": 5,
               "GeopoliticalEntity": 6, "Person": 7, "Artifact-Generic": 8, "AspatialInformationStore": 9,
               "ConceptualWork": 10, "Animal": 11, "GeographicalRegion": 12, "Place-NonAgent": 13, "Group": 14,
               "CommercialOrganization": 15, "InformationBearingThing": 16, "Plant": 17, "FictionalThing": 18,
               "IntelligentAgent": 19, "Thing": 20, "Action": 21, "NonProfitOrganization": 22, "Weapon": 23,
               "InanimateObject-Natural": 24, "MathematicalOrComputationalThing": 25, "InorganicMaterial": 26,
               "ConstructionArtifact": 27, "BiologicalLivingObject": 28, "OrganismPart": 29, "ChemicalObject": 30,
               "EdibleStuff": 31, "OrganicMaterial": 32, "ColoredThing": 33, "Food": 34, "Microorganism": 35,
               "Drink": 36, "TimeInterval": 37, "Product": 38, "GeometricFigure": 39}

for k,v in labels_dict.items():
    labels_dict[k]=v-1

print('Found %s labels.' % len(labels_dict))

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu', border_mode='same')(embedded_sequences)

# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu', border_mode='same')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu', border_mode='same')(x)

# x = MaxPooling1D(10)(x)
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)


x = GRU(128, return_sequences=False, consume_less='gpu', dropout_W=0.0, dropout_U=0.0)(embedded_sequences)

preds = Dense(len(labels_dict), activation='sigmoid')(x)

model = Model(sequence_input, preds)

print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
print(model.summary())
# happy learning!
# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#           nb_epoch=2, batch_size=128)

# for a in Xy_generator(pad_generator(
#     batch_generator(
#         data_generator(train_path),
#         batch_size=256), tokenizer, labels_dict, MAX_SEQUENCE_LENGTH)):
#     # print(len(a))
#     # print(len(a[0]))
#     # print(len(a[0][0]))
#     # print(len(a[0][0][0]))
#     # print(a)
#     break



test_data = []
for x in pad_generator(
    batch_generator(
        data_generator(test_path),
        batch_size=3509), tokenizer, labels_dict, MAX_SEQUENCE_LENGTH):
    test_data.extend(x)
    break
test_data2 = [x for x in Xy_generator([test_data])][0]


callbacks = [LossHistory(Evaluator(test_data2, labels_dict))]

model.fit_generator(Xy_generator(pad_generator(
    batch_generator(
        data_generator(train_path),
        batch_size=256), tokenizer, labels_dict, MAX_SEQUENCE_LENGTH)),
    samples_per_epoch=10000, nb_epoch=100, validation_data=test_data2, nb_val_samples=3509, callbacks=callbacks)  # 2799601
