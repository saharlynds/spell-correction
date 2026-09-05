import xlrd
from django.template import Template, Context
from django.http import HttpResponse
import os
from . import rule_checkspell

# Optional word-substitution lookup, currently disabled in secondpage() below.
# The spreadsheet has two columns: column 0 is the word to match, column 1 is
# its replacement. To re-enable it, fill in your own path here and uncomment the
# xlrd block in secondpage(). Left commented out so the view has no unused
# module-level state while the feature is off.
# WORD_SPELLCHECK_XLSX = ""   # e.g. WordsSpellCheck.xlsx

# The neural corrector is optional. It needs two artifacts, neither of which is
# committed (both live in the gitignored datasets/ directory):
#
#   bi1000.h5                 trained weights from network_architecture.py
#   english-german-both.pkl   the cleaned corpus from neural_network.py, needed
#                             to rebuild the exact tokenizers the model was
#                             trained with
#
# When either is missing or fails to load, the view falls back to the rule
# corrector and prints why. See the README for how to produce them.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
MODEL_FILE = os.path.join(DATA_DIR, 'bi1000.h5')
BOTH_PKL = os.path.join(DATA_DIR, 'english-german-both.pkl')

# Populated on first request by _load_neural(). _neural_checked makes the load
# attempt happen once per process rather than once per request, so a missing
# model does not re-log on every page view.
_neural = None
_neural_checked = False


def _log(message):
    print('[spell-correction] ' + message)


def _load_neural():
    """Load the trained model and its tokenizers, or return None.

    Returns a dict with the model, both tokenizers and the source sequence
    length, or None when the neural path is unavailable for any reason. Never
    raises: every failure is reported and turns into a rule based fallback.

    network_prediction.py cannot be imported for this - it runs its whole
    evaluation at module level - so the encode/decode steps below deliberately
    mirror its predict_sequence() rather than calling into it.
    """
    global _neural, _neural_checked
    if _neural_checked:
        return _neural
    _neural_checked = True

    missing = [p for p in (MODEL_FILE, BOTH_PKL) if not os.path.exists(p)]
    if missing:
        _log('no trained model available (missing: ' + ', '.join(missing) + ')')
        _log('falling back to the rule corrector')
        return None

    try:
        from pickle import load
        from keras.models import load_model
        from keras.preprocessing.text import Tokenizer
        from spell_correction_ui.Capsule_Keras import Capsule

        with open(BOTH_PKL, 'rb') as handle:
            dataset = load(handle)

        def make_tokenizer(lines):
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(lines)
            return tokenizer

        # Column 0 is the correct/target side, column 1 the erroneous input -
        # the same convention neural_network.py writes and
        # network_architecture.py trains on.
        target_tokenizer = make_tokenizer(dataset[:, 0])
        source_tokenizer = make_tokenizer(dataset[:, 1])
        source_length = max(len(line.split()) for line in dataset[:, 1])

        model = load_model(MODEL_FILE, custom_objects={'Capsule': Capsule})
    except Exception as exc:
        _log('failed to load the neural model from ' + MODEL_FILE +
             ' (' + type(exc).__name__ + ': ' + str(exc) + ')')
        _log('falling back to the rule corrector')
        return None

    _log('loaded neural model from ' + MODEL_FILE)
    _neural = {
        'model': model,
        'target_tokenizer': target_tokenizer,
        'source_tokenizer': source_tokenizer,
        'source_length': source_length,
    }
    return _neural


def neural_correct(sentence):
    """Correct *sentence* with the trained model, or return None to fall back."""
    bundle = _load_neural()
    if bundle is None:
        return None

    try:
        from numpy import argmax
        from keras.preprocessing.sequence import pad_sequences

        encoded = bundle['source_tokenizer'].texts_to_sequences([sentence])
        encoded = pad_sequences(encoded, maxlen=bundle['source_length'],
                                padding='post')
        prediction = bundle['model'].predict(encoded, verbose=0)[0]

        index_to_word = {i: w for w, i
                         in bundle['target_tokenizer'].word_index.items()}
        words = []
        for vector in prediction:
            word = index_to_word.get(int(argmax(vector)))
            if word is None:
                break
            words.append(word)
        return ' '.join(words)
    except Exception as exc:
        _log('neural prediction failed (' + type(exc).__name__ + ': ' +
             str(exc) + '); falling back to the rule corrector')
        return None


def correct(sentence):
    """Correct *sentence*, preferring the neural model when it is usable."""
    corrected = neural_correct(sentence)
    if corrected:
        return corrected
    return rule_checkspell.rule_checkspell(sentence.split())


def secondpage(request):
    if 'input' in request.GET and request.GET['input']:
        now = request.GET['input']  # datetime.datetime.now()
        # from 1 to len()-2
        a = now.split()
        # Simple way of using templates from the filesystem.
        # This is BAD because it doesn't account for missing files!
        # loc = WORD_SPELLCHECK_XLSX  # set at the top of this file
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
        word2 = correct(joint_word1)
        fp = open(os.path.join('spell_correction_ui', 'templates', 'second.html'), encoding='utf-8')
        t = Template(fp.read())
        fp.close()
        html = t.render(Context({'current_data': now, 'process_data': word2}))
        return HttpResponse(html)
    else:
        # Simple way of using templates from the filesystem.
        # This is BAD because it doesn't account for missing files!
        # fp = open('spell_correction_ui/templates/second.html', encoding='utf-8')
        fp = open(os.path.join('spell_correction_ui', 'templates', 'second.html'), encoding='utf-8')
        t = Template(fp.read())
        fp.close()
        html = t.render(Context({'process_data': ''}))
        return HttpResponse(html)
