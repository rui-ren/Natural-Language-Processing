# Sentiment analysis

# with the rise of online social media platforms like Twitter, Facebook, and Reddit, and the proliferation of customer reviews on Amazon and Yelp.
# we have access, more than ever before, to massive text-based datasets.  We can use this dataset to determine how large protions of population feel about the products.


import os
import glob
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS

def read_imdb_data(data_dir='data/imdb-reviews'):

    """
    - Data/
        - train/
            - pos/
            - neg/
        - test/
            - pos/
            - neg/
    """
    
    # data, labels to return in nested dicts matching the dir structure
    
    data = {}
    labels = {}
    
    # iterate the train and test datafile
    
    for data_type in ['train', 'test']:
        # use the two dictionary --> collections.default(lambda defaultdict(str))
        data[data_type] = {}
        labels[data_type] = {}
        
        # restore different sentiment analysis
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            # fetch list of file for the sentiment
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            # find the dataset
            files = glob.glob(path)
            
            # read rewiew data and put in the corpus
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    data[data_type][sentiment].append(sentiment)
            
            assert len(data[data_type][sentiment] == len(labels[data_type][sentiment]), +
                    '{}/{} data size does not match labels size'.format(data_type, sentiment)
    
    return data, labels
    
    
def review_wordcloud(sentiment):
    """
    @ sentiment type: we want to see
    """
    combined_text = ' '.join([review for review in data['train'][sentiment]])
    
    # initialize word cloud object
    wc = WordCloud(background_colors = 'white', max_words = 50, stopwords = STOPWORDS.update(['br', 'film', 'movie']))
    
    # Generate and plot worldcloud
    plt.imshow(wc.generate(combined_text))
    plt.axis('off')
    plt.show()
    
    
if __name__ == "__main__":
    data, labels = read_imdb_data()
    print("IMDb reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
        len(data['train']['pos']), len(data['train']['neg']),
        len(data['test']['pos']), len(data['test']['neg'])))
    
    print(data['train']['pos'][2])
    print(len(data['train']['pos']), 'The number of positive sentiment in training set')
    
    # generate the wordCloud
    
        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
            