from logits_to_text import *
# test the model performance

def final_model(x, y, x_tk, y_tk):
    
    tmp_x = pad(preproc_english_sentences)
    model = model_final( tmp_x.shape,
                preproc_french_sentences.shape[1],
                len(english_tokenizer.word_index)+1,
                len(french_tokenizer.word_index)+1
    )

    print(logits_to_text(model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=17, validation_split=0.2)))

    

