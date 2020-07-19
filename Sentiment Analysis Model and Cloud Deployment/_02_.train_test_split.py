## preparation for the training set and test set

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import chain

def prepare_imdb_data(data):
    """
    @ transfer to the pandas dataframe
    """
    data = pd.DataFrame(data)
    
    # combine the positive and negative reviews and labels
    df = []
    
    # get the train words
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']
    
    # random shuffle the data
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    
    return data_train, data_test, labels_train, labels_test
    
