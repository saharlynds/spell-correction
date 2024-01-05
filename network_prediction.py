from pickle import load
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from Capsule_Keras import *


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
    w1 = open("/content/drive/MyDrive/formann.txt","w", encoding='utf-8')
    w2 = open("/content/drive/MyDrive/inform1.txt","w", encoding='utf-8')
    w3 = open("/content/drive/MyDrive/formtrue.txt","w", encoding='utf-8')
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
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')
vtest = load_clean_sentences('valid.pkl')
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
r1 = open("/content/drive/MyDrive/informalAnn.txt", encoding='utf-8')
f1 = r1.readlines()
validation = list()
for k in f1:
    validation.append(k)
r1.close()
validationx = encode_sequences(ger_tokenizer, ger_length, validation)
modeltrain = '/content/drive/MyDrive/bi1000.h5'
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