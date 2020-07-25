
# Deep Neural Networks for Acoustic Modeling

from keras import backend as K
from keras.models import Model 
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)


# Model 1: 
    """" Simple model only use RNN model """

def simple_rnn_model(input_dim, output_dim=29):
    """
    Build a recurrent network for speech
    """