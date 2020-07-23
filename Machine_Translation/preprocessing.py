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


# load english word -- ETL
english_sentence = helper.load_data('/data/small_vocab_en')
# load french word -- ETL
french_sentence = helper.load_data('/data/small_vocab_fr')

# vocabulary -- BoW
english_words_counter = collections.Counter([word for sentence in english_sentence for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentence for word in sentence.split()])


for sample_i in range(2):
    print("small_vocab_en Line {}: {}".format(sample_i+1, english_sentence[sample_i]))
    print("small_vocab_fr Line {}: {}".format(sample_i+1, french_sentence[sample_i]))
    

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    
    x_tk = Tokenizer(char_level=False)
    x_tk.fit_on_texts(x)
    
    return x_tk.texts_to_sequences(x), x_tk
    
    
def pad(x, length=None):
    
    """
    pad x
    : param x: List of sequence
    : param length: Length to pad the sequence to. If None, use the length of longest sequence in x_tk
    : return: Padded numpy array of sequences
    """
    if not length:
        length = max([len(sequence) for sequence in x])
    
    return pad_sequences(x, max_len=length, padding='post)


def preprocessing(x, y):
    """
    Preprocessing x and y
    : param x: Feature List of sentence
    : param y: Label List of sentences/strings
    : return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    # tokenizer
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    
    # padding
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    
    # Keras sparse categorical crossentropy --> 3 x 1
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    
    return preprocess_x, preprocess_y, x_tk, y_tk
    
    
    
    
    
    
    