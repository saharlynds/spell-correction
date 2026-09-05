from pickle import load
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import os
import sys

# Run directly (python spell_correction_ui/network_prediction.py) or as a
# module. Direct execution puts this file's own directory on sys.path, not the
# repo root, so the spell_correction_ui.* import below needs this first.
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spell_correction_ui.Capsule_Keras import *

# Every data file and model artifact lives in spell_correction_ui/datasets/,
# which is gitignored - nothing under it is committed. Paths resolve relative to
# this file, so the script runs from any working directory. See the README for
# how to produce each input.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

# Inputs from neural_network.py.
BOTH_PKL = os.path.join(DATA_DIR, 'english-german-both.pkl')
TRAIN_PKL = os.path.join(DATA_DIR, 'english-german-train.pkl')
TEST_PKL = os.path.join(DATA_DIR, 'english-german-test.pkl')
VALID_PKL = os.path.join(DATA_DIR, 'valid.pkl')

# The trained weights from network_architecture.py. Point this at the checkpoint
# you want to evaluate.
MODEL_FILE = os.path.join(DATA_DIR, 'bi1000.h5')

# Optional extra validation input, one sentence per line. Left unset by default;
# fill in a path to evaluate it via the evaluate_model call at the bottom.
VALIDATION_FILE = os.path.join(DATA_DIR, 'informalAnn.txt')

# Outputs, written side by side and line-aligned: the model predictions, the
# erroneous source sentences, and the gold targets. These are the three files
# obtaining_evaluation_quantities__test.py reads.
PREDICTED_OUT = os.path.join(DATA_DIR, 'formann.txt')
SOURCE_OUT = os.path.join(DATA_DIR, 'inform1.txt')
GOLD_OUT = os.path.join(DATA_DIR, 'formtrue.txt')

for _path, _made_by in ((BOTH_PKL, 'neural_network.py'),
                        (TRAIN_PKL, 'neural_network.py'),
                        (TEST_PKL, 'neural_network.py'),
                        (VALID_PKL, 'neural_network.py'),
                        (MODEL_FILE, 'network_architecture.py')):
    if not os.path.exists(_path):
        raise SystemExit(
            'Missing required file: ' + _path +
            ' -- produce it with ' + _made_by + ' (see the README).'
        )


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


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    #print(source)
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
    w1 = open(PREDICTED_OUT, 'w', encoding='utf-8')
    w2 = open(SOURCE_OUT, 'w', encoding='utf-8')
    w3 = open(GOLD_OUT, 'w', encoding='utf-8')
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, eng_tokenizer, source)
        # print(translation)
        # print(raw_dataset[i])
        raw_target, raw_src = raw_dataset[i]

        # print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        w1.write(translation + '\n')
        w2.write(raw_src.split('\n')[0] + '\n')
        w3.write(raw_target + '\n')
        # actual.append([raw_target.split()])
        # predicted.append(translation.split())
    w1.close()
    w2.close()
    w3.close()
    # calculate BLEU score
    # print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    # print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    # print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    # print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# evaluate the skill of the model
def evaluate_model1(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, eng_tokenizer, source)
        raw_src = raw_dataset[i]
        # print(translation)
        # print(raw_dataset[i])
        if i < 10:
            print('src=[%s], predicted=[%s]' % (raw_src, translation))
        predicted.append(translation.split())


# load datasets
dataset = load_clean_sentences(BOTH_PKL)
train = load_clean_sentences(TRAIN_PKL)
test = load_clean_sentences(TEST_PKL)
vtest = load_clean_sentences(VALID_PKL)
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
# prepare data
# trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
# testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
vtestX = encode_sequences(ger_tokenizer, ger_length, vtest[:, 1])
a = list()
a.append("است سابت من رندگی حمه")
a.append("هسمت قریب شره این در")
a.append("حملح بدح ادامح")
x1 = encode_sequences(ger_tokenizer, ger_length, a)

# load model
r1 = open(VALIDATION_FILE, encoding='utf-8')
f1 = r1.readlines()
validation = list()
for k in f1:
    validation.append(k)
r1.close()
validationx = encode_sequences(ger_tokenizer, ger_length, validation)
modeltrain = MODEL_FILE
model = load_model(modeltrain, custom_objects={'Capsule': Capsule})
# model = load_model('model.h5')
# test on some training sequences
# evaluate_model1(model, eng_tokenizer, x1, a)
print('train')
# evaluate_model(model, eng_tokenizer, validationx, validation)
evaluate_model(model, eng_tokenizer, vtestX, vtest)
# evaluate_model(model, eng_tokenizer, trainX, train)
# test on some test sequences
print('test')
# evaluate_model(model, eng_tokenizer, testX, test)