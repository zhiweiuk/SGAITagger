'''Text chunking dataset for testing sequence labeling architectures.

Source: http://www.cnts.ua.ac.be/conll2002
'''

from __future__ import absolute_import, unicode_literals
from six.moves import cPickle
import gzip
from keras.utils.data_utils import get_file
from six.moves import zip
import numpy as np
import sys
import os
from collections import Counter
from itertools import chain
import codecs

POS_TAGS = [
    ".",
    "\"",
    ",",
    "(",
    ")",
    ":",
    "-X-",
    "$",
    "''",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "NN|SYM",
    "PDT",
    "POS",
    "PRP$",
    "PRP",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP$",
    "WP",
    "WRB",
]

CHUNK_TAGS = [
    "-X-",
    "B-ADJP",
    "B-ADVP",
    "B-NP",
    "B-PP",
    "B-SBAR",
    "B-VP",
    "I-ADJP",
    "I-ADVP",
    "I-CONJP",
    "I-INTJ",
    "I-LST",
    "I-NP",
    "I-PP",
    "I-PRT",
    "I-SBAR",
    "I-VP",
    "O",
]

NER_TAGS = [
    'B-LOC',
    'I-LOC',
    'B-MISC',
    'I-MISC',
    'B-ORG',
    'I-ORG',
    'B-PER',
    'I-PER',
    'O',
]

def load_data(word_preprocess=lambda x: x):
    '''Loads the conll2002 Named Entity Recognition dataset.

    # Arguments:
        word_preprocess: A lambda expression used for filtering the word forms.
            For example, use `lambda w: w.lower()` when all words should be
            lowercased.
    '''
    X_words_train, X_pos_train, X_chunk_train, y_train = load_file('data/conll2003/eng.train')
    X_words_test, X_pos_test, X_chunk_test, y_test = load_file('data/conll2003/eng.testb')

    index2word = _fit_term_index(X_words_train, reserved=['<PAD>', '<UNK>'], preprocess=word_preprocess)
    word2index = _invert_index(index2word)

    index2pos = POS_TAGS
    pos2index = _invert_index(index2pos)

    index2chunk = CHUNK_TAGS
    chunk2index = _invert_index(index2chunk)

    index2ner = NER_TAGS
    ner2index = _invert_index(index2ner)

    X_words_train = np.array([[word2index[word_preprocess(w)] for w in words] for words in X_words_train])
    X_pos_train = np.array([[pos2index[t] for t in pos_tags] for pos_tags in X_pos_train])
    X_chunk_train = np.array([[chunk2index[t] for t in chunk_tags] for chunk_tags in X_chunk_train])
    y_train = np.array([[ner2index[t] for t in ner_tags] for ner_tags in y_train])

    X_words_test = np.array([[word2index.get(word_preprocess(w), word2index['<UNK>']) for w in words] for words in X_words_test])
    X_pos_test = np.array([[pos2index[t] for t in pos_tags] for pos_tags in X_pos_test])
    X_chunk_test = np.array([[chunk2index[t] for t in chunk_tags] for chunk_tags in X_chunk_test])
    y_test = np.array([[ner2index[t] for t in ner_tags] for ner_tags in y_test])

    return (X_words_train, X_pos_train, X_chunk_train, y_train), (X_words_test, X_pos_test, X_chunk_test, y_test), (index2word, index2pos, index2chunk, index2ner)


def load_file(filename):
    '''Loads and parses a conll2002 data file.

    # Arguments:
        filename: The requested filename.
        md5_hash: The expected md5 hash.
    '''
    with codecs.open(filename, encoding='latin-1') as fd:
        rows = _parse_grid_iter(fd)
        words, pos_tags, chunk_tags, ner_tags = zip(*[zip(*row) for row in rows])
    return words, pos_tags, chunk_tags, ner_tags


def _parse_grid_iter(fd, sep=' '):
    '''
    Yields the parsed sentences for a given file descriptor
    '''
    sentence = []
    for line in fd:
        if line == '\n' and len(sentence) > 0:
            yield sentence
            sentence = []
        else:
            sentence.append(line.strip().split(sep))
    if len(sentence) > 0:
        yield sentence

def _fit_term_index(terms, reserved=[], preprocess=lambda x: x):
    all_terms = chain(*terms)
    all_terms = map(preprocess, all_terms)
    term_freqs = Counter(all_terms).most_common()
    id2term = reserved + [term for term, tf in term_freqs]
    return id2term


def _invert_index(id2term):
    return {term: i for i, term in enumerate(id2term)}