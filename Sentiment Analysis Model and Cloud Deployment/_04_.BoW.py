# Look at the Compute Bag of Words feature
# The file we will generate the vocabulary

import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import sklearn.preprocessing as pr

# def BoW model here

def extract_BoW_features(words_train,
                        words_test,
                        vocabulary_size=5000,
                        cache_dir=cache_dir,
                        cache_dir="bow_features.pkl"):
    
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), 'rb') as f:
                cache_data = joblib.load(f)
        except:
            pass
    
    if cache_data is None:
        
        # generate a CounterVectorizer object and fit the data
        
        vectorizer = CountVectorizer(max_features=vocabulary_size, preprocessor=lambda x: x, tokenizer=lambda x: x)
        features_train = vectorizer.fit_transform(words_train).toarray()
        features_test = vectorizer.fit_transform(words_test).toarry()
        
        # pickle to file
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_ 
            cache_data = dict(features_train=feature_train, features_test=features_test, vocabulary=vocabulary)
            
            # save to pickle file
            with open(os.path.join(cache_dir, cache_file), 'wb') as f:
                joblib.dump(cache_data, f)
        
        else:
            features_train, features_test, vocabulary = (cache_data["features_train"], cache_data["features_data"], cache_data["vocabulary"])
        
        return features_train, features_test, vocabulary

def normalize(data_train, data_test):
    feature_train = pr.normalize(data_train, axis=1)
    feature_test = pr.normalize(data_test,axis=1)

if __name__ == "__main__":

    features_train, features_test, vocabulary = extract_BoW_features(words_train, words_test)
    print("Sample words: {}".format(random.sample(len(vocabulary), 8)))
    
    print("\n -------- Preprocessed words")
    print(words_train[5])
    print("\n -------- Bag of Words features -----")
    print(features_train[5])
    print("\n -----Label------")
    print(labels_train[5])
    
    plt.plot(features_train[5, :])
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.show()
    
    
    
    
        