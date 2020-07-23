
# define logits to sequence

def logits_to_text(logits, tokenizer):
    # mapping from the probability to french words
    """
    Turn logits from a neural network into text using tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = "<PAD>"

    # check the largest numpy value in the argument here!
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

