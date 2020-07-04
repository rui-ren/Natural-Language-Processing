# dataloader
from helper import *

import matplotlib.pyplot as plt
import numpy as np

from itertools import chain
from collections import Counter, defaultdict
from helper import show_model, Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution


"""
API for data:
data.X : word sequences
data.Y : tag sequences 
data.stream: method returns an iterator that chains together every pair of (word, tag) for the corpus
"""

if __name__ == "__main__":
    # Load the dataset
    data = Dataset('tags-universal.txt', 'brown-universal.txt', train_test_split=0.8)
    print("There are {} sentences in the corpus.".format(len(data)))
    print("There are {} sentences in the training set.".format(len(data.training_set)))
    print("There are {} sentences in the testing set.".format(len(data.testing_set)))
    
    for i in range(2):
        print("Sentence {}:".format(i+1), data.X[i], '\n')
        print("Sentence {}:".format(i+1), data.Y[i], '\n')
    
    print("\nStream (word, tag) pairs: \n")
    for i, pair in enumerate(data.stream()):
        print("\t", pair)
        if i > 5: break
    