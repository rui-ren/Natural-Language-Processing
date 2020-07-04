# Most Frequent Count Baseline Model
# this is the MFC baseline model for the calculation

"""
Here we will build the Most Frequent Count 
This is the baseline for the HMM model

    * The most simply baseline for tagger performance is to use tag frequently assigned to each word 
"""
from collections import defaultdict
from collections import namedtuple

FakeState = namedtuple("FakeState", "name")
# MFC table:
class MFCTagger:
    # Note:
    missing = FakeState(name="<MISSING>")
    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in table.items()})

    def viterbi(self, seq):
        """ Integrate the Pomegrante viberbi API"""
        """ Build the Markov Chain here"""
        return 0., list(enumerate(["<start>" + [self.table[w] for w in seq] + ["<end>"]


def pair_count(tags, words):
    """
    @tags: 
        the tags sentences
    @words:
        the words sentences
    
    @return: 
        --> dictionary type
        {Noun: {
            Natural: 12,
            Language: 23,
            ...
        }, Verb {},...
        }
    """
    count = defaultdict(lambda : defaultdict(int))
    for tag, word in zip(tags, words):
        count[count][word] += 1
    return count


def replace_unknown(sequence):
    """
    replace the word with unknown nan if word not in frozen wordset
    """
    return [w if w in data.training_set.voca else 'nan' for w in sequence]
    
    
def simplify_decoding(X, model):
    """
    X should be a 1- D sequence of observation for the model to predict
    """
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]]
 
 
def accuracy(X, Y, model):
    """
    Calculate the prediction accuracy with the true lable
    """
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions
    

if __name__ == "__main__":
    tags = [tag for i, (word, tag) in enumerate(data.training_set.stream())]
    words = [word for i, (word, tag) in enumerate(data.training_set.stream())]
    # emission count: mapping from words ---> tag  --->  p(tag|word)
    emission_count = pair_count(tags, words)
    
    assert len(emission_count) == 12, "There should be 12 tags in the unversal text"
    assert max(emission_count["NOUN"], key=emission_count["NOUN"].get) == "time", "Hmmm...time is supposed to be the most common NOUN"
    
    # example for decoding sequences with MFC Tagger
    for key in data.testing_set.keys[:3]:
        print( "Sentence Key: {}\n".format(key))
        print("Predicted labels: \n-------------")
        print(simplify_decoding(data.sentences[key].words, mfc_model))
        print('\n')
        print(data.sentences[key].tags)
        print('\n')
        
    # test the MFCTagger for the calculation
    # transition : p(word|tag)
    word_counts = pair_count(words, tags)
    # select the maximum tag here!
    mfc_table = dict((word, max(tags.keys(), key = lambda key: tags[key]) for word, tags in word_counts.items()))
    # get the mfc table
    mfc_model = MFCTagger(mfc_table)
    
    assert len(mfc_table) == len(data.training_set.vocab), ""
    assert all(k in data.training_set.vocab for k in mfc_table.keys()), ""
    mfc_training_acc = accuracy(data.training_set.X, data.training_set.Y, mfc_model)
    print("training accuracy mfc_model: {:.2f}%".format(100 * mfc_training_acc))

    mfc_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, mfc_model)
    print("testing accuracy mfc_model: {:.2f}%".format(100 * mfc_testing_acc))
        
        
        
        
        
        
        
        
        
        
        
        


