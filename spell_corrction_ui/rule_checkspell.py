from hazm import InformalNormalizer, Lemmatizer, Stemmer
import xlrd
from spell_corrction_ui.levenshtein import levenshtein
from spell_corrction_ui.character_rules.key_transfer import rules_keytransfer
from spell_corrction_ui.character_rules.insert_character import rules_insert_character
from spell_corrction_ui.character_rules.repetition_charachter import rules_repetition_character
from spell_corrction_ui.character_rules.delete_character import rules_delete
from spell_corrction_ui.character_rules.symphonious_character import rules_symphonious
from spell_corrction_ui.character_rules.displacement_character import rules_displacement
import os


def rule_checkspell(words):
    loc = os.path.join('spell_corrction_ui', 'datasets', 'databasecorect.txt')
    # To open Workbook
    # wb = xlrd.open_workbook(loc)
    # sheet = wb.sheet_by_index(0)
    sli = []
    normalizer = InformalNormalizer()
    lemmatizer = Lemmatizer()
    stemmer = Stemmer()
    set1 = []
    with open(loc, "r") as file:
        lines = file.readlines()
        for line in lines:
            set1.append(line)
    # for i in range(sheet.nrows):
    #     set1.append(sheet.cell_value(i, 0))
    # set1 = set(arr)
    for j1 in words:
        text_after1 = normalizer.normalized_word(j1)
        lem1 = lemmatizer.lemmatize(j1).split('#')
        stem1 = stemmer.stem(j1)
        jj = 0
        if (text_after1 in set1) or (lem1[0] in set1) or (stem1 in set1) or (j1 in set1):
            jj = 1
            sli.append(j1)
        text_after = []
        if jj == 0:
            wor1 = rules_symphonious(j1)
            for x in wor1:
                norm2 = normalizer.normalized_word(x)
                lem2 = lemmatizer.lemmatize(x).split('#')
                stem2 = stemmer.stem(x)
                if x in set1 or lem2[0] in set1 or stem2 in set1 or norm2 in set1:
                    text_after.append(x)
            wor2 = rules_displacement(j1)
            for x in wor2:
                norm2 = normalizer.normalized_word(x)
                lem2 = lemmatizer.lemmatize(x).split('#')
                stem2 = stemmer.stem(x)
                if x in set1 or lem2[0] in set1 or stem2 in set1 or norm2 in set1:
                    text_after.append(x)
            wor3 = rules_keytransfer(j1)
            for x in wor3:
                norm2 = normalizer.normalized_word(x)
                lem2 = lemmatizer.lemmatize(x).split('#')
                stem2 = stemmer.stem(x)
                if x in set1 or lem2[0] in set1 or stem2 in set1 or norm2 in set1:
                    text_after.append(x)
            wor4 = rules_repetition_character(j1)
            for x in wor4:
                norm2 = normalizer.normalized_word(x)
                lem2 = lemmatizer.lemmatize(x).split('#')
                stem2 = stemmer.stem(x)
                if x in set1 or lem2[0] in set1 or stem2 in set1 or norm2 in set1:
                    text_after.append(x)
            wor5 = rules_delete(j1)
            for x in wor5:
                norm2 = normalizer.normalized_word(x)
                lem2 = lemmatizer.lemmatize(x).split('#')
                stem2 = stemmer.stem(x)
                if x in set1 or lem2[0] in set1 or stem2 in set1 or norm2 in set1:
                    text_after.append(x)
            wor6 = rules_insert_character(j1)
            for x in wor6:
                norm2 = normalizer.normalized_word(x)
                lem2 = lemmatizer.lemmatize(x).split('#')
                stem2 = stemmer.stem(x)
                if x in set1 or lem2[0] in set1 or stem2 in set1 or norm2 in set1:
                    text_after.append(x)
            unique_list = []
            # unique_list.append('(')
            # unique_list.append(j1)
            for x in text_after:
                # check if exists in unique_list or not
                if x not in unique_list:
                    unique_list.append(x)
            unique_list.append(j1)
            levensht_ar = []
            for x in unique_list:
                min_ = 100000
                for ij in set1:
                    x1 = levenshtein(x, ij)
                    if x1 < min_:
                        min_ = x1
                levensht_ar.append([x, min_])
            levensht_ar1 = sorted(levensht_ar, key=lambda levensht_ar: levensht_ar[1])
            # unique_list.append(')')
            levensht_ar12 = []
            for x in levensht_ar1:
                levensht_ar12.append(x[0])
            sss = levensht_ar12[0]  # '|'.join(levensht_ar12)#levensht_ar12[1]
            sli.append(sss)
    s1 = ' '.join(sli) + "\n"
    return s1
