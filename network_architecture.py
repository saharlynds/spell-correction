from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from matplotlib import pyplot as plt
from keras.layers import Dropout
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import BatchNormalization
from Capsule_Keras import *
import numpy as np
import xlwt
from xlwt import Workbook
import pandas as pd
from keras.callbacks import EarlyStopping
# import fasttext
# import visualkeras
from sklearn.manifold import TSNE
import random


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X
    # one hot encode target sequence


def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        # encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(sequence)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1])
    return y
    # define NMT model


def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units, embedding_matrix):
    # model.add(Embedding(src_vocab, 300, weights = [embedding_matrix], input_length = src_timesteps, trainable = True))
    # model.add(Conv1D(4 * n_units, 3, activation='relu', padding='same'))
    # model.add(MaxPooling1D(pool_size=2, padding='same'))
    # model.add(Capsule(num_capsule=10, dim_capsule=16, routings=3, share_weights = True))
    # model.add(Bidirectional(LSTM(n_units)))
    model = Sequential()
    # model.add(Embedding(src_vocab, 100, input_length=src_timesteps, mask_zero = True))
    model.add(Embedding(src_vocab, 300, weights=[embedding_matrix], input_length=src_timesteps, trainable=True))
    # model.add(Conv1D(100, 3, activation='relu', padding='same'))
    # model.add(MaxPooling1D(pool_size=2, padding='same'))
    # model.add(Capsule(num_capsule=50, dim_capsule=50, routings=3, share_weights = True))
    model.add(Bidirectional(LSTM(1000, dropout=0.5)))
    model.add(RepeatVector(tar_timesteps))
    # model.add(Bidirectional(LSTM(n_units,dropout=0.5)))
    # model.add(RepeatVector(tar_timesteps))
    model.add(Bidirectional(LSTM(1000, return_sequences=True, dropout=0.5)))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    # visualkeras.layered_view(model)
    return model


# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
# print(eng_tokenizer.word_index.keys())
# print(eng_tokenizer.word_index)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
# print(ger_tokenizer.word_index)
# print(ger_tokenizer.word_index)
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))
# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)
# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)
print('finish stup')
# embedding_matrix=pre_train_glove(ger_vocab_size,word_index)
embedding_matrix = np.loadtxt('embedding_matrix.txt')
# define model
hi = []
wb = Workbook()
www = '/content/drive/MyDrive/' + 'data.xlsx'
writer1 = pd.ExcelWriter(www)
df1 = pd.DataFrame()
df1.to_excel(writer1, sheet_name='x1')
writer1.save()
writer1.close()
for i in range(1, 2):
    writer = pd.ExcelWriter(www, engine='openpyxl', mode='a')
    model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 1000, embedding_matrix)
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
    # summarize defined model
    print(model.summary())
    # plot_model(model, to_file='model.png', show_shapes=True)
    # keras.utils.plot_model(model, "m1.png", show_shapes=True)
    # fit model
    filename = '/content/drive/MyDrive/' + 'model' + str(i) + '.h5'
    checkpoint = ModelCheckpoint(filename, monitor='acc', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    history = model.fit(trainX, trainY, epochs=2000, batch_size=64, validation_data=(testX, testY),
                        callbacks=[checkpoint, es], verbose=2)
    # strcom = str(i)+'.txt'
    # sheet1 = wb.add_sheet(strcom)
    # w2 = open(strcom,'w', encoding='utf-8')
    df = pd.DataFrame(history.history)
    df.to_excel(writer, sheet_name=str(i))
    # for i in range(len(history.history['acc'])):
    #    #w2.write(str(history.history['acc'][i])+'
    #    '+str(history.history['val_acc'][i])+' '+str(history.history['loss'][i])+'
    #    '+str(history.history['val_loss'][i])+'\n')
    #    # sheet1.write(0, 0, 'ISBT DEHRADUN')
    # w2.close()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    str1 = 'accuracy' + str(i) + '.png'
    str2 = 'loss' + str(i) + '.png'
    plt.savefig(str1)
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(str2)
    plt.show()
    writer.save()
    writer.close()
