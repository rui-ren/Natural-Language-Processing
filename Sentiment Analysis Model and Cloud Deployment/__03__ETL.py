# ETL
import os
from bs4 import BeautifulSoup
import read
import nltk
from nltk import word_tokenize
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
import pickle
# we use beautifulSoup, it can easily ETL for Javascript

def review_to_words(review):
    """
    @ remove  the HTML tags and non-letter,
        - convert to lower case, normalization
        - remove punctuation
        - stemming
        - lemmatisation
    """
    soup = BeautifulSoup(review, 'html5lib')
    sentence = soup.get_text().lower()
    sentence = re.sub(r"[a-zA-Z0-9]", ' ', sentence)
    sentence = sentence.split()
    words = [char for char in sentence if char not in stopwords('english')]
    return words

cache_dir = os.path.join("cache", "sentiment_analysis")
os.makedirs(cache_dir, exist_ok=True)

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file='preprocessed_data.pkl'):
                    
    """ Convert each review to words, read from cache if available"""
    cache_data = None
    
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), 'rb') as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass
    
    # if cache file is missing
    if cache_data is None:
        word_train = list(map(review_to_words, data_train))
        word_test = list(map(review_to_words, data_test))
        
        # we still need to put in the pickle file
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test, labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), 'wb') as f:
                pickle.dump(cache_data, f)
            print('Wrote preprocessed data to cache file: ', cache_file)
        
    else:
        # unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],,
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test
    

if __name__ == "__main__":
    words_train, words_test, labels_train, labels_test = preprocess_data(
        data_train, data_test, labels_train, label_test)
            
    print("\n---- Raw review ---")
    print(data_train[1])
    print("\n --- Preprocessing words ----")
    print(words_train[1])
    print("\n---Label---")
    print(labels_train[1])
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    