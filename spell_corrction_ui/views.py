import xlrd
from django.template import Template, Context
from django.http import HttpResponse
import os
from . import rule_checkspell


def secondpage(request):
    if 'input' in request.GET and request.GET['input']:
        now = request.GET['input']  # datetime.datetime.now()
        # from 1 to len()-2
        a = now.split()
        # Simple way of using templates from the filesystem.
        # This is BAD because it doesn't account for missing files!
        # loc = r"C:\Users\saman\PycharmProjects\untitled\aslidatasiroos\WordsSpellCheck.xlsx"
        # To open Workbook
        # wb = xlrd.open_workbook(loc)
        # sheet = wb.sheet_by_index(0)
        word1 = []
        for word in a:
            jj = 0
            # for i in range(sheet.nrows):
            #     if str(sheet.cell_value(i, 0)) == str(word):
            #         jj = 1
            #         word1.append(str(sheet.cell_value(i, 1)))
            #         break
            if jj == 0:
                word1.append(word)
        joint_word1 = ' '.join(word1)
        word2 = rule_checkspell.rule_checkspell(joint_word1.split())
        # sentencAnn = ann_siroos(now)
        fp = open(os.path.join('spell_corrction_ui', 'templates', 'second.html'), encoding='utf-8')
        t = Template(fp.read())
        fp.close()
        # html = t.render(Context({'current_data': now, 'process_data': word2, 'process_data1': sentencAnn[0]}))
        html = t.render(Context({'current_data': now, 'process_data': word2}))
        return HttpResponse(html)
    else:
        # Simple way of using templates from the filesystem.
        # This is BAD because it doesn't account for missing files!
        # fp = open('spell_corrction_ui/templates/second.html', encoding='utf-8')
        fp = open(os.path.join('spell_corrction_ui', 'templates', 'second.html'), encoding='utf-8')
        t = Template(fp.read())
        fp.close()
        html = t.render(Context({'process_data': ''}))
        return HttpResponse(html)
