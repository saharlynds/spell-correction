from django.template import Template, Context
from django.http import HttpResponse
from django.core.management import execute_from_command_line
from hazm import InformalNormalizer, Lemmatizer, Stemmer
import xlrd
from levenshtein import levenshtein
from character_rules.key_transfer import rules_keytransfer
from character_rules.insert_character import rules_insert_character
from character_rules.repetition_charachter import rules_repetition_character
from character_rules.delete_character import rules_delete
from character_rules.symphonious_character import rules_symphonious
from character_rules.displacement_character import rules_displacement


def rule_checkspell(words):
    loc = "datasets/databasecorect.xlsx"
    # To open Workbook
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sli = []
    normalizer = InformalNormalizer()
    lemmatizer = Lemmatizer()
    stemmer = Stemmer()
    set1 = []
    for i in range(sheet.nrows):
        set1.append(sheet.cell_value(i, 0))
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


def secondpage(request):
    if 'input' in request.GET and request.GET['input']:
        now = request.GET['input']  # datetime.datetime.now()
        # from 1 to len()-2
        a = now.split()
        # Simple way of using templates from the filesystem.
        # This is BAD because it doesn't account for missing files!
        loc = r"C:\Users\saman\PycharmProjects\untitled\aslidatasiroos\WordsSpellCheck.xlsx"
        # To open Workbook
        wb = xlrd.open_workbook(loc)
        sheet = wb.sheet_by_index(0)
        word1 = []
        for word in a:
            jj = 0
            for i in range(sheet.nrows):
                if str(sheet.cell_value(i, 0)) == str(word):
                    jj = 1
                    word1.append(str(sheet.cell_value(i, 1)))
                    break
            if jj == 0:
                word1.append(word)
        joint_word1 = ' '.join(word1)
        word2 = rule_checkspell(joint_word1.split())
        # sentencAnn = ann_siroos(now)
        fp = open('second.html', encoding='utf-8')
        t = Template(fp.read())
        fp.close()
        # html = t.render(Context({'current_data': now, 'process_data': word2, 'process_data1': sentencAnn[0]}))
        html = t.render(Context({'current_data': now, 'process_data': word2}))
        return HttpResponse(html)
    else:
        # Simple way of using templates from the filesystem.
        # This is BAD because it doesn't account for missing files!
        fp = open('second.html', encoding='utf-8')
        t = Template(fp.read())
        fp.close()
        html = t.render(Context({'process_data': ''}))
        return HttpResponse(html)
