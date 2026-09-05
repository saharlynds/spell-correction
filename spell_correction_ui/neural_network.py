from google.colab import drive
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
from hazm import *

drive.mount('/content/drive')


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


def clean_pairs1(lines):
    cleaned = list()
    for pair in lines:
        clean_pair = list()
        for line in pair:
            st1 = re.sub(r'[a-zA-Z:1923456789..!؟><?123456789.]+', r'', line)
            # st1=line.replace("؟",(""
            normalizer = Normalizer()
            a = normalizer.normalize(st1)
            clean_pair.append(a)
        cleaned.append(clean_pair)
    return array(cleaned)


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# load dataset
filename = '/content/drive/MyDrive/pairsiroos.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# print(pairs)
# clean sentences
clean_pairs = clean_pairs1(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-german.pkl')
# spot check
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))

# New Section
from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# load dataset
raw_dataset = load_clean_sentences('english-german.pkl')
# reduce dataset size
n_sentences = 1000000
dataset = raw_dataset[:n_sentences, :]
# random shuffle
# shuffle(dataset)
# split into train/test
train, test = dataset[:800000], dataset[800000:1000000]
valid = raw_dataset[1000000:1190000, :]

# save
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')
save_clean_data(valid, 'valid.pkl')
