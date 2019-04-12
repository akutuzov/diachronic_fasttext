#!/bin/env python3
# coding: utf-8

import sys
import gensim
import logging
import zipfile
import json

# Simple toy script to get an idea of what one can do with word embedding models using Gensim
# Models can be found at http://vectors.nlpl.eu/explore/embeddings/models/
# or in the /projects/nlpl/data/vectors/latest/ directory on Abel

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    embeddings_file = sys.argv[1]  # File containing word embeddings

    logger.info('Loading the embedding model...')

    # Detect the model format by its extension:

    # Binary word2vec format:
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')
    # Text word2vec format:
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')
    # ZIP archive from the NLPL vector repository:
    elif embeddings_file.endswith('.zip'):
        with zipfile.ZipFile(embeddings_file, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open('meta.json')
            metadata = json.loads(metafile.read())
            for key in metadata:
                print(key, metadata[key])
            print('============')

            # Loading the model itself:
            stream = archive.open("model.bin")  # or model.txt, if you want to look at the model
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors='replace')
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(embeddings_file)

        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)

    emb_model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)

    logger.info('Finished loading the embedding model...')

    logger.info('Model vocabulary size: %d' % len(emb_model.vocab))

    logger.info('Example of a word in the model: "%s"' % emb_model.index2word[3])

    while True:
        query = input("Enter your word (type 'exit' to quit):")
        if query == "exit":
            exit()
        words = query.strip().split()
        # If there's only one query word, produce nearest associates
        if len(words) == 1:
            word = words[0]
            print(word)
            if word in emb_model:
                print('=====')
                print('Associate\tCosine')
                for i in emb_model.most_similar(positive=[word], topn=10):
                    print(i[0]+'\t', i[1])
                print('=====')
            else:
                print('%s is not present in the model' % word)

        # Else, find the word which doesn't belong here
        else:
            print('=====')
            print('This word looks strange among others:', emb_model.doesnt_match(words))
            print('=====')

