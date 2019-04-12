#!python

import logging
import multiprocessing
import sys
import gensim
from gensim.models import FastText

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# How many workers (CPU cores) to use during the training?
cores = multiprocessing.cpu_count()  # Use all cores we have access to
logger.info('Number of cores to use: %d' % cores)

corpus = sys.argv[1]
skipgram = int(sys.argv[2])
window = int(sys.argv[3])
mincount = int(sys.argv[4])
iterations = 1
logger.info(corpus)

data = gensim.models.word2vec.LineSentence(corpus)

filename = corpus.replace('.txt.gz', '_ft') + '_' + str(skipgram) + '_' + str(window) + '.model'

model = FastText(data, size=100, sg=skipgram, min_count=mincount, window=window, min_n=2, max_n=5,
                 hs=0, negative=3, workers=cores, iter=iterations, seed=42, bucket=2000000)
model.save(filename)
