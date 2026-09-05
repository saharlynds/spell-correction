# Persian Spell Correction

Automatic spell correction for Persian (Farsi) text: a rule based corrector
built on [hazm](https://github.com/roshan-research/hazm), plus a neural
sequence to sequence model (fastText embeddings + Keras/TensorFlow), served
through a Django web UI.

The view uses the neural model when trained weights are present, and falls back
to the rule based corrector otherwise. Weights are not committed, so a fresh
clone uses the rule based path.

## Data

The correct Persian sentences come from the
[Tatoeba project](https://tatoeba.org), licensed
[CC-BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/):

> Sentences sourced from the Tatoeba Project (<https://tatoeba.org>), used under
> CC-BY 2.0 FR.

The misspelled sentences are not a separate dataset. `generate_error_pairs.py`
produces them from those same sentences using the six rules in
`character_rules/`:

| Module | What it does |
|---|---|
| `symphonious_character.py` | Swaps homophonous letters (ه/ح, ت/ط, ع/ا, غ/ق) |
| `displacement_character.py` | Swaps letters adjacent on the Persian keyboard |
| `key_transfer.py` | Transposes two adjacent characters |
| `repetition_charachter.py` | Doubles a character |
| `delete_character.py` | Deletes a character |
| `insert_character.py` | Inserts a keyboard neighbour character |

`rule_checkspell.py` uses the same six rules in reverse, to generate correction
candidates for a misspelled word.

### Files you provide

Data and weights are not committed. All of them go in
`spell_correction_ui/datasets/`:

| File | What it is |
|---|---|
| `databasecorect.txt` | Correct Persian sentences, one per line, from Tatoeba |
| `pairs.txt` | Training pairs, from `generate_error_pairs.py` |
| `cc.fa.300.bin` | Persian fastText vectors, from <https://fasttext.cc> |
| `bi1000.h5` | Trained weights, from `network_architecture.py` |

## Install and run

Requires Python 3.11.

```bash
pip install -r requirements.txt
python manage.py runserver
```

Open <http://127.0.0.1:8000/secondpage/> and enter a Persian phrase. Run the
server from the repository root.

Keras is pinned to the 2.x line because the code uses `keras.preprocessing`,
which Keras 3 removed. `SECRET_KEY` and `DEBUG = True` are development defaults;
change both before any real deployment.

## What each file is for

| File | Purpose |
|---|---|
| `rule_checkspell.py` | Rule based corrector |
| `levenshtein.py` | Edit distance, used to rank candidates |
| `character_rules/` | The six error rules |
| `generate_error_pairs.py` | Builds the training pairs |
| `neural_network.py` | Cleans the corpus and splits train/test/valid |
| `word_embedding.py` | Builds the fastText embedding matrix |
| `network_architecture.py` | Defines and trains the model |
| `network_prediction.py` | Inference |
| `obtaining_evaluation_quantities__test.py` | Accuracy, precision, recall, F1, BLEU |
| `Capsule_Keras.py` | Capsule layer, experimental |
| `views.py`, `urls.py`, `settings.py` | Django project |

Run the training scripts in the order they appear above, from
`generate_error_pairs.py` through `network_prediction.py`.

## License

Code: GPL-3.0, see [LICENSE](LICENSE).

Sentence data: Tatoeba, <https://tatoeba.org>, licensed CC-BY 2.0 FR. Not
distributed in this repository.
