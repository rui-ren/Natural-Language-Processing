# # 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, TimeDistributed, Dense, Activation, RepeatVector, Bidirectional
from keras.optimizer import Adam
from keras.layers.embeddings import Embedding
from Keras.models import Model, Sequential
from keras.loss import sparse_categorical_crossentropy

# MODEL 1:  SIMPLE RNN model

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """

    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return Keras build, but not trained
    
    """


    learning_rate = 1e-3
    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)

    model = Model(input_seq, Activation('softmax')(logits))

    model.compile(loss=sparse_categorical_crossentropy,
                optimizer=Adam(learning_rate),
                metrics=['accuracy']
    )

    return model


def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train RNN model using word embedding on x and y
    """
    learning_rate = 1e-3
    rnn = GRU(64, return_sequences=True, activation='tanh')
    embedding = Embedding(french_vocab_size, 64, input_length=input_shape[1])
    logits = TimeDistributed(Dense(french_vocab_size, activation='softmax'))

    model = Sequential()
    model.add(embedding)
    model.add(rnn)
    model.add(logits)
    model.compile(loss=sparse_categorical_crossentropy,
                optimizer=Adam(learning_rate),
                metrics=['accuracy']
            )


def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):

    """
    Build and train bidirectional RNN model on x and y
    """
    learning_rate = 1e-3
    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences=True, dropout=0.1),\
            input_shape=input_shape[1:]
        ))

    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy
            optimizer=Adam(learning_rate),
            metrics=['accuracy']
            )
    return model


def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train and encoder-decoder model on x and y
    """
    learning_rate = 1e-3
    model = Sequential()
    model.add(GRU(128, input_shape=input_shape[1:], return_sequences=False))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(128, return_sequences=True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy,
            optimizer=Adam(learning_rate),
            metrics=['accuracy']
            )
    return model


def final_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build the embedding - encoder decoder - bidirectional model
    """
    model = Sequnetial()
    model.add(Embedding(input_dim=english_vocab_size, output_dim=128, input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256, return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
    learning_rate = 0.005
    model.compile(loss=sparse_categorical_crossentropy,
                optimizer=Adam(learning_rate),
                metrics=['accuracy']
            )
    return model


if __name__ == "__main__":
    
    # input data
    tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
    tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

    # train the neural network
    simple_rnn_model = simple_model(
        tmp_x.shape,
        max_french_sequence_length,
        english_vocab_size,
        french_vocab_size
    )


    # build the model
    simple_rnn_model.fit(
        tmp_x,
        preproc_french_sentences,
        batch_size=1024,
        epochs=10,
        validation_split=0.2
    )

    print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))
    