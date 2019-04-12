#!python

import numpy as np
import logging
import sys
import gensim
from gensim.models.utils_any2vec import compute_ngrams, ft_ngram_hashes
from gensim import utils


def save_word2vec(fname, vocab, vectors, binary=False):
    """Store the weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.
    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else in plain text.
.
    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    vector_size = vectors.shape[1]
    logging.info("storing %sx%s projection weights into %s", len(vocab), vector_size, fname)
    with utils.smart_open(fname, 'wb') as fout:
        fout.write(utils.to_utf8("%s %s\n" % (len(vocab), vector_size)))
        for el in sorted(vocab.keys()):
            row = vectors[vocab[el]]
            if binary:
                row = row.astype(np.float32)
                fout.write(utils.to_utf8(el) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (el, ' '.join(repr(val) for val in row))))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    modelfile = sys.argv[1]
    freq_threshold = int(sys.argv[2])
    filename = sys.argv[3]

    model = gensim.models.FastText.load(modelfile)
    model.init_sims(replace=True)

    print(model)

    ngram_identifiers = {}
    hashes = set()

    for word in model.wv.vocab:
        human_ngrams = compute_ngrams(word, model.min_n, model.max_n)
        hash_ngrams = ft_ngram_hashes(word, model.min_n, model.max_n, model.bucket)
        for hum, hsh in zip(human_ngrams, hash_ngrams):
            if hum not in ngram_identifiers:
                ngram_identifiers[hum] = {}
                ngram_identifiers[hum]['hash'] = hsh
                ngram_identifiers[hum]['freq'] = 1
            ngram_identifiers[hum]['freq'] += 1
            hashes.add(hsh)

    print('Unique ngram hashes:', len(hashes))
    print('Unique ngrams:', len(ngram_identifiers))

    fin_ngram_identifiers = {n: ngram_identifiers[n]['hash']
                             for n in ngram_identifiers
                             if ngram_identifiers[n]['freq'] > freq_threshold}

    hashes = [fin_ngram_identifiers[i] for i in fin_ngram_identifiers]
    print('Number of ngrams:', len(hashes))
    print('Number of unique hashes:', len(set(hashes)))

    # hashes = sorted(list(hashes))
    # ngram_arr = model.wv.vectors_ngrams[hashes, :]
    # print(ngram_arr.shape)

    save_word2vec(filename, fin_ngram_identifiers, model.wv.vectors_ngrams)
