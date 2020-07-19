## 

# end to end machine learning pipeline
# Translate the machine learning pipeline from 

import collections

import helper
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Models
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

# load english word
english_sentence = helper.load_data('/data/small_vocab_en')
# load french word
french_sentence = helper.load_data('/data/small_vocab_fr')

# vocabulary
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentence for word in sentence.split()])


