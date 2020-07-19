# Gaussian Naive Baysian

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

def Gassian_NB(features_train, labels_test):

    clf1 = GaussianNB()
    clf1.fit(features_train, labels_train)

    print("[{}] Accuracy: train = {}, test = {}".format(
            clf1.__class__.__name__,
            clf1.score(features_train, labels_train),
            clf1.score(features_test, labels_test)
            ))
 
def classify_gboost(X_train, X_test, y_train, y_test, n_estimators=32):
    # initialize classifier
    
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0, max_depth=1, random_state=0)
    
    # classifier
    clf.fit(X_train, y_train)
    
    print("[{}] Accuracy: train = {}, test = {}".format(
        clf1.__class__.__name__,
        clf1.score(features_train, labels_train),
        clf1.score(features_test, labels_test)
        ))
    # return the classifier model
    return clf

if __name__ == "__main__":
    
    clf1 = Gassian_NB(features_train, labels_train)
    
    my_review = "I thought this it would be just like all the other boring sequals that coming out everyday, horrible"
    true_sentiment = "pos"
    
    my_words = review_to_words(my_review)
    vectorizer = CountVectorizer(vocabulary=vocabulary,
                                preprocessor=lambda x: x, tokenizer=lambda x: x)
    my_bow_features = vectorizer.transform([my_words]).toarray()
    
    clf2 = classify_gboost(X_train, X_test, y_train, y_test)
    
    print("\nTrue sentiment: {}, predicted sentiment: {}".format(true_sentiment, predicted_sentiment))
    # 
    