'''Named Entity Recognition dataset for testing sequence labeling architectures.

Source: http://www.cnts.ua.ac.be/conll2003
'''

from __future__ import absolute_import, unicode_literals
from six.moves import cPickle
import gzip
from keras.utils.data_utils import get_file
from six.moves import zip
import numpy as np
import sys
import os
import re
from collections import Counter
from itertools import chain
import codecs
from colors import green, red, blue

# Preprocessing config
FLATTEN_NUMBERS = True
REMOVE_IOB = False
IOBES_NER_SCHEME = True
IOBES_CHUNK_SCHEME = True
POS_PREPROCESS = True
GAZ = False

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

BASE_CHUNK_TAGS = [
    "ADJP",
    "ADVP",
    "NP",
    "PP",
    "SBAR",
    "VP",
    "CONJP",
    "INTJ",
    "LST",
    "PRT",
]

CHUNK_TAGS = list()

BASE_NER_TAGS = [
    "LOC",
    "PER",
    "MISC",
    "ORG",
]

NER_TAGS = list()

PUNC = [".", "\"", ",", "(", ")", ":", "-X-", "$", "''"]

# converts a number to 0
# 2342 --> 0, 99 --> 0, 'peter' --> 'peter'
def flatten_number(x):
    return re.sub(r'\d+', '0', x)

# turns iob tags into iobes tags
def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag == '-X-': # we'll also turn -X- into O
            new_tags.append('O')
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

# Removes the iob tag
def remove_iob(tags):
    """
    Removes IOB tagging scheme
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag == '-X-': # we'll also turn -X- into O
            new_tags.append('O')
        elif "-" in tag:
            new_tags.append(tag.split("-")[1])
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

# Makes -X- (Docstarts) into an O tag
def normalize_x(tags):
    new_tags = list()
    for i, tag in enumerate(tags):
        if tag == '-X-':
            new_tags.append('O')
        else:
            new_tags.append(tag)
    return new_tags

# removes the punctuation tags
def normalize_pos_tags(tags):
    new_tags = list()
    for i, tag in enumerate(tags):
        if tag in PUNC: # we'll also turn -X- into O
            new_tags.append('O')
        else:
            new_tags.append(tag)
    return new_tags

# loads and preprocesses the data
# it is also responsible for creating the embedding matrix
def load_data(word_preprocess=lambda x: x):
    '''Loads the conll2003 Named Entity Recognition dataset.

    # Arguments:
        word_preprocess: A lambda expression used for filtering the word forms.
            For example, use `lambda w: w.lower()` when all words should be
            lowercased.
    '''
    print(blue("Preprocessor: Loading in CoNLL data"))
    X_words_train, X_pos_train, X_chunk_train, y_train = load_file('data/conll2003/eng.train')
    X_words_dev, X_pos_dev, X_chunk_dev, y_dev = load_file('data/conll2003/eng.testa')
    X_words_test, X_pos_test, X_chunk_test, y_test = load_file('data/conll2003/eng.testb')

    # gaz = load_gaz("misc")
    # file = open("misc.gaz","w")
    # for i in gaz:
    #     for j in i:
    #         for k in j:
    #             if(len(k.split(" ")) == 4):
    #                  file.write(k + "\n")
    #         file.write("\n")
    # file.close()

    if GAZ == True:
        print(blue("Preprocessor: Loading in gazetteers"))
        per_words_train, per_pos_train, per_chunk_train, per_train = load_file('data/gazetteers/per.gaz')
        org_words_train, org_pos_train, org_chunk_train, org_train = load_file('data/gazetteers/org.gaz')
        loc_words_train, loc_pos_train, loc_chunk_train, loc_train = load_file('data/gazetteers/loc.gaz')
        misc_words_train, misc_pos_train, misc_chunk_train, misc_train = load_file('data/gazetteers/misc.gaz')

        print(blue("Preprocessor: Merging gazetteers into training set"))
        X_words_train = X_words_train + per_words_train + org_words_train + loc_words_train + misc_words_train
        X_pos_train = X_pos_train + per_pos_train + org_pos_train + loc_pos_train + misc_pos_train
        X_chunk_train = X_chunk_train + per_chunk_train + org_chunk_train + loc_chunk_train + misc_chunk_train
        y_train = y_train + per_train + org_train + loc_train + misc_train

    if FLATTEN_NUMBERS == True:
        print(blue("Preprocessor: Transforming all numbers to 0"))
        X_words_train = tuple([[flatten_number(w) for w in words] for words in X_words_train])
        X_words_dev = tuple([[flatten_number(w) for w in words] for words in X_words_dev])
        X_words_test = tuple([[flatten_number(w) for w in words] for words in X_words_test])


    ##
    ## NER PREPROCESS
    ##
    if REMOVE_IOB == False:
        UNIQUE_NER_TAGS = list()
        # Creates the IOBES Tags for the y_truth
        if IOBES_NER_SCHEME == True:
            print(blue("Preprocessor: Using IOBES tagging scheme for NER tags"))
            iobes = ["I", "B", "E", "S"]
            y_train = tuple([iob_iobes(word) for word in y_train])
            y_dev = tuple([iob_iobes(word) for word in y_dev])
            y_test = tuple([iob_iobes(word) for word in y_test])

            for i in BASE_NER_TAGS:
                for j in iobes:
                    UNIQUE_NER_TAGS.append(j + "-" + i)
        else:
            print("Preprocessor: Using IOB tagging scheme for NER tags")
            iob = ["I", "B"]
            for i in BASE_NER_TAGS:
                for j in iob:
                    UNIQUE_NER_TAGS.append(j + "-" + i)

        UNIQUE_NER_TAGS.append("O")

        print(green("Unique NER Tags:"), UNIQUE_NER_TAGS)
        NER_TAGS = UNIQUE_NER_TAGS
    else:
        print("Preprocessor: Removing IOB tagging scheme from NER tags")
        y_train = tuple([remove_iob(word) for word in y_train])
        y_dev = tuple([remove_iob(word) for word in y_dev])
        y_test = tuple([remove_iob(word) for word in y_test])
        NER_TAGS = BASE_NER_TAGS
        NER_TAGS.append("O")

    ##
    ## CHUNK PREPROCESS
    ##

    UNIQUE_CHUNK_TAGS = list()
    UNIQUE_CHUNK_TAGS.append("O")

    if IOBES_CHUNK_SCHEME == True:
        print(blue("Preprocessor: Using IOBES tagging scheme for CHUNK tags"))
        iobes = ["I", "B", "E", "S"]
        X_chunk_train = tuple([iob_iobes(word) for word in X_chunk_train])
        X_chunk_dev = tuple([iob_iobes(word) for word in X_chunk_dev])
        X_chunk_test = tuple([iob_iobes(word) for word in X_chunk_test])

        for i in BASE_CHUNK_TAGS:
            for j in iobes:
                UNIQUE_CHUNK_TAGS.append(j + "-" + i)
    else:
        print(blue("Preprocessor: Using IOB tagging scheme for CHUNK tags"))
        iobes = ["I", "B"]

        X_chunk_train = tuple([normalize_x(word) for word in X_chunk_train])
        X_chunk_dev = tuple([normalize_x(word) for word in X_chunk_dev])
        X_chunk_test = tuple([normalize_x(word) for word in X_chunk_test])

        for i in BASE_CHUNK_TAGS:
            for j in iobes:
                UNIQUE_CHUNK_TAGS.append(j + "-" + i)

    print(green("Unique Chunk Tags:"), UNIQUE_CHUNK_TAGS)
    CHUNK_TAGS = UNIQUE_CHUNK_TAGS

    ##
    ## Preprocess POS tags
    ##

    if POS_PREPROCESS == True:
        print(blue("Preprocessor: Preprocessing POS Tags"))
        X_pos_train = tuple([normalize_pos_tags(word) for word in X_pos_train])
        X_pos_dev = tuple([normalize_pos_tags(word) for word in X_pos_dev])
        X_pos_test = tuple([normalize_pos_tags(word) for word in X_pos_test])
        POS_TAGS.append('O')

    index2word = _fit_term_index(X_words_train + X_words_dev, reserved=['<PAD>', '<UNK>'], preprocess=word_preprocess)
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

    X_words_dev = np.array([[word2index[word_preprocess(w)] for w in words] for words in X_words_dev])
    X_pos_dev = np.array([[pos2index[t] for t in pos_tags] for pos_tags in X_pos_dev])
    X_chunk_dev = np.array([[chunk2index[t] for t in chunk_tags] for chunk_tags in X_chunk_dev])
    y_dev = np.array([[ner2index[t] for t in ner_tags] for ner_tags in y_dev])

    X_words_test = np.array([[word2index.get(word_preprocess(w), word2index['<UNK>']) for w in words] for words in X_words_test])
    X_pos_test = np.array([[pos2index[t] for t in pos_tags] for pos_tags in X_pos_test])
    X_chunk_test = np.array([[chunk2index[t] for t in chunk_tags] for chunk_tags in X_chunk_test])
    y_test = np.array([[ner2index[t] for t in ner_tags] for ner_tags in y_test])

    embeddings_index = {}
    f = open(os.path.join('data/glove/', 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print(blue('Found %s word vectors.' % len(embeddings_index)))

    embedding_matrix = np.zeros((len(word2index) + 1, 100)) #100 = embedding dimension
    for word, i in word2index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print(green("Preprocessor: I'm finished preprocessing :-)"))

    return (X_words_train, X_pos_train, X_chunk_train, y_train), (X_words_dev, X_pos_dev, X_chunk_dev, y_dev), (X_words_test, X_pos_test, X_chunk_test, y_test), (index2word, index2pos, index2chunk, index2ner), (embedding_matrix)

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

def load_gaz(gaz_type):
    filename = "data/gazetteers/" + gaz_type + ".raw"
    gaz = list()
    TAG = gaz_type.upper()
    with open(filename, 'r') as f:
        l = [line.strip() for line in f]
    for i in l:
        sentence_list = []
        stack = []
        if len(i.split(" ")) > 1:
            words = i.split(" ")
            for word in words:
                if word == words[0]:
                    stack.append(word + " NNP" + " " + "I-NP" + " " + "B-" + TAG)
                else:
                    stack.append(word + " NNP" + " " + "I-NP" + " " + "I-" + TAG)
        else:
            stack.append(i.split(" ")[0] + " NNP" + " " + "I-NP" + " " + "B-" + TAG)
        sentence_list.append(tuple(stack))
        gaz.append(sentence_list)
    return gaz

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
