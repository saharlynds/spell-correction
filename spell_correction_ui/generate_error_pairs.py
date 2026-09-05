"""Build the synthetic parallel corpus used to train the neural corrector.

Reads a file of correct Persian sentences (one per line), injects spelling
errors into each one by applying a random subset of the character_rules
operations, and writes a tab-separated pairs file:

    <correct sentence>\t<sentence with injected errors>

That column order is what neural_network.py expects: to_pairs() splits each line
on a tab, and network_architecture.py then feeds column 1 to the model as input
and trains against column 0 as the target.

The base sentences come from the Tatoeba project (https://tatoeba.org), licensed
CC-BY 2.0 FR. The erroneous side is generated here, not downloaded.

Usage:
    python spell_correction_ui/generate_error_pairs.py
    python spell_correction_ui/generate_error_pairs.py --errors-per-sentence 2
    python spell_correction_ui/generate_error_pairs.py --variants 3 --seed 7
"""
import argparse
import os
import random
import sys

# Run directly (python spell_correction_ui/generate_error_pairs.py) or as a
# module (python -m spell_correction_ui.generate_error_pairs). Direct execution
# puts this file's own directory on sys.path, not the repo root, so the package
# imports below would fail without this.
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spell_correction_ui.character_rules.delete_character import rules_delete
from spell_correction_ui.character_rules.displacement_character import rules_displacement
from spell_correction_ui.character_rules.insert_character import rules_insert_character
from spell_correction_ui.character_rules.key_transfer import rules_keytransfer
from spell_correction_ui.character_rules.repetition_charachter import rules_repetition_character
from spell_correction_ui.character_rules.symphonious_character import rules_symphonious

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

# Input: correct Persian sentences, one per line (the Tatoeba extract).
DEFAULT_SENTENCES = os.path.join(DATA_DIR, 'databasecorect.txt')

# Output: the tab-separated pairs file neural_network.py reads.
DEFAULT_PAIRS = os.path.join(DATA_DIR, 'pairs.txt')

# The six error operations, in the order they appear in character_rules/. Each
# takes a word and returns a deduplicated list of variants, the first of which
# is usually the unmodified word, so callers must filter out no-op results.
RULES = (
    ('symphonious', rules_symphonious),
    ('displacement', rules_displacement),
    ('key_transfer', rules_keytransfer),
    ('repetition', rules_repetition_character),
    ('delete', rules_delete),
    ('insert', rules_insert_character),
)


def corrupt_word(word, rng):
    """Return *word* with one rule applied, or None if no rule changed it.

    Rules are tried in random order so no single operation dominates the corpus.
    A rule that only returns the input unchanged (common for short words, where
    the index-based rules fall through) is skipped.
    """
    for _, rule in rng.sample(RULES, len(RULES)):
        variants = [v for v in rule(word) if v and v != word]
        if variants:
            return rng.choice(variants)
    return None


def corrupt_sentence(sentence, rng, errors_per_sentence):
    """Inject up to *errors_per_sentence* word-level errors into *sentence*.

    Returns None when nothing could be corrupted, so the caller can drop the
    sentence rather than emit a pair whose two sides are identical.
    """
    words = sentence.split()
    if not words:
        return None

    # Only words the rules can actually alter are eligible, so a sentence of
    # short/unalterable words is dropped instead of silently yielding a no-op
    # pair. Candidate order is shuffled so errors are spread across the sentence.
    positions = list(range(len(words)))
    rng.shuffle(positions)

    changed = 0
    for i in positions:
        if changed >= errors_per_sentence:
            break
        corrupted = corrupt_word(words[i], rng)
        if corrupted is not None:
            words[i] = corrupted
            changed += 1

    if changed == 0:
        return None
    return ' '.join(words)


def generate(sentences_path, pairs_path, errors_per_sentence, variants, seed):
    rng = random.Random(seed)

    with open(sentences_path, encoding='utf-8') as handle:
        sentences = [line.strip() for line in handle]
    sentences = [s for s in sentences if s]

    if not sentences:
        raise SystemExit('No sentences found in ' + sentences_path)

    os.makedirs(os.path.dirname(pairs_path) or '.', exist_ok=True)

    written = 0
    skipped = 0
    with open(pairs_path, 'w', encoding='utf-8', newline='\n') as out:
        for sentence in sentences:
            # A tab in the source would corrupt the two-column format.
            correct = sentence.replace('\t', ' ')
            emitted = 0
            for _ in range(variants):
                erroneous = corrupt_sentence(correct, rng, errors_per_sentence)
                if erroneous is None:
                    continue
                out.write(correct + '\t' + erroneous + '\n')
                emitted += 1
            written += emitted
            if emitted == 0:
                skipped += 1

    print('Read %d sentences from %s' % (len(sentences), sentences_path))
    print('Wrote %d pairs to %s' % (written, pairs_path))
    if skipped:
        print('Skipped %d sentences no rule could alter' % skipped)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sentences', default=DEFAULT_SENTENCES,
                        help='input file of correct sentences, one per line '
                             '(default: %(default)s)')
    parser.add_argument('--out', default=DEFAULT_PAIRS,
                        help='output tab-separated pairs file '
                             '(default: %(default)s)')
    parser.add_argument('--errors-per-sentence', type=int, default=1,
                        help='max word-level errors to inject per sentence '
                             '(default: %(default)s)')
    parser.add_argument('--variants', type=int, default=1,
                        help='erroneous variants to emit per input sentence '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed, fixed so runs are reproducible '
                             '(default: %(default)s)')
    args = parser.parse_args(argv)

    if not os.path.exists(args.sentences):
        raise SystemExit(
            'Missing input sentences: ' + args.sentences + '\n'
            'Download the Persian sentences from https://tatoeba.org/en/downloads '
            'and save them one per line at that path - see the README.'
        )
    if args.errors_per_sentence < 1:
        raise SystemExit('--errors-per-sentence must be at least 1')
    if args.variants < 1:
        raise SystemExit('--variants must be at least 1')

    generate(args.sentences, args.out, args.errors_per_sentence,
             args.variants, args.seed)


if __name__ == '__main__':
    main()
