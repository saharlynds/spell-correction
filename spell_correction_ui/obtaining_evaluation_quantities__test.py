from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

# Fill in the paths to your own evaluation output files from Step 6
# (network_prediction.py). Each is one sentence per line, and all three must be
# line-aligned: line N of every file must describe the same sentence.
PREDICTED_FILE = ""   # model's predicted corrections (e.g. formann.txt)
SOURCE_FILE = ""      # the erroneous input sentences (e.g. inform1.txt)
GOLD_FILE = ""        # the correct/gold sentences (e.g. formtrue.txt)

_unset = [name for name, path in (('PREDICTED_FILE', PREDICTED_FILE),
                                  ('SOURCE_FILE', SOURCE_FILE),
                                  ('GOLD_FILE', GOLD_FILE)) if not path]
if _unset:
    raise SystemExit(
        'Set PREDICTED_FILE/SOURCE_FILE/GOLD_FILE at the top of this script '
        'before running. Still empty: ' + ', '.join(_unset)
    )

# Keep these assignments in this order. Downstream the three handles are used by
# role, not by name: f3 (gold) is the BLEU reference, f2 (predicted) is the BLEU
# hypothesis, and f1 (source) only answers "was this token already correct?" in
# the tp/tn/fp/fn block. Swapping the source and predicted files silently
# inverts tp with fn and scores BLEU against the input instead of the output.
r1 = open(SOURCE_FILE, encoding='utf-8')
r2 = open(PREDICTED_FILE, encoding='utf-8')
r3 = open(GOLD_FILE, encoding='utf-8')
f1 = r1.readlines()
f2 = r2.readlines()
f3 = r3.readlines()
tp = 0
tn = 0
fp = 0
fn = 0
l1 = 0
for i in range(0, len(f1)):
    a1 = f1[i].split('\n')[0].split(' ')
    a2 = f2[i].split('\n')[0].split(' ')
    a3 = f3[i].split('\n')[0].split(' ')
    l1 = l1 + len(a3)
    max1 = min(len(a3), len(a2), len(a1))
    # while len(a3)<max1:
    #   # a3.append(' ')
    # while len(a2)<max1:
    #   # a2.append(' ')
    # while len(a1)<max1:
    #   # a1.append(' ')
    for j in range(0, max1):
        if a3[j] == a1[j] and a3[j] == a2[j]:
            tn = tn + 1
        elif a3[j] == a1[j] and a3[j] != a2[j]:
            fn = fn + 1
        elif a3[j] != a1[j] and a3[j] == a2[j]:
            tp = tp + 1
        elif a3[j] != a1[j] and a3[j] != a2[j]:
            fp = fp + 1
print('tp: %f' % tp)
print('tn: %f' % tn)
print('fp: %f' % fp)
print('fn: %f' % fn)
ftotal = tp + tn + fp + fn
print('toralTpFpTnFn: %f' % ftotal)
print('totalwords: %f' % l1)
accuracy = (tp+tn) / (tp+tn+fp+fn)
perfection = (tp) / (tp + fp)
recall = (tp) / (tp+fn)
f_measure = (2 * perfection * recall) / (perfection + recall)
print("accuracy:")
print(accuracy)
print("perfection:")
print(perfection)
print("recall:")
print(recall)
print("f_measure:")
print(f_measure)
actual, predicted = list(), list()
actual1 = list()
for i in range(0,len(f1)-1):
    actual.append([f3[i].split('\n')[0].split(' ')])
    predicted.append(f2[i].split('\n')[0].split(' '))
    actual1.append(f3[i].split('\n')[0].split(' '))
print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0, 1.0, 0, 0)))
print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0, 0, 1.0, 0)))
print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
print('BLEUTotal: %f' % corpus_bleu(actual, predicted))
array1 = []
actual2 = ''
predicted2 = ''
for i in range(0, len(actual1)):
    max2 = min(len(actual1[i]), len(predicted[i]))
    while len(actual1[i]) > max2:
        actual1[i].pop()
    while len(predicted[i]) > max2:
        predicted[i].pop()
    for ii in range(0, len(actual1[i])):
        actual2 = actual2 + ' ' + actual1[i][ii]
        predicted2 = predicted2 + ' ' + predicted[i][ii]
    a = precision_recall_fscore_support(actual1[i], predicted[i], average='micro')
    array1.append(a)
kk = 0
per = 0
re = 0
f = 0
print("test")
print(len(array1))
for i in range(500):
    # print(array1[i])
    per = per + array1[i][0]
    re = re + array1[i][1]
    f = f + array1[i][2]
    kk = kk + 1
print(per / kk)
print(re / kk)
print(f / kk)
print(precision_recall_fscore_support(actual2.split(' '), predicted2.split(' '), average='micro'))
print(accuracy_score(actual2.split(' '), predicted2.split(' ')))
