# Here we switch the gear from traditional machine learning to RNN
# Keras has the IMDB dataset

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

def RNN_model(X_train, X_test, vocabulary=5000, embedding_size=32, max_words=500):

    # pad sequence
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)
    # design RNN for sentiment analysist

    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_lenght=max_words))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
    print("Load dataset with {} training samples, {} test samples".format(len(X_train), len(X_test)))
    print("----Review----")
    print(X_train[7])
    print("---Label---")
    print(y_train[7])

    word2id = imbd.get_word_index()
    id2word = {i: word for word, i in word2.id.items()}
    print("---Review (with words) ----")
    print([id2word.get(i, "") for i in X_train[7]])
    print("---Label---")
    print(y_train[7])
    
    batch_size = 64
    num_epochs = 3
    # here we define the batch size for the testing
    X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
    X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]
    
    model.fit(X_train2, y_train2, validation_data = (X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
    # save model here
    model_file = "rnn_model.h5"
    model.save(os.path.join(cache_dir, model_file))
    
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", scores[1])
    
    

