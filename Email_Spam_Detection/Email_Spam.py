# Email Spam Detection using Naive Bayesian Model
"""
# Naive Bayesian model will give us a look up table, it is lazy learning method
# If we have categorical data and numerical data, we need us numpy.hstack(categorical, numerical)
# Guassian naive baysian or Multinomial bayesian, Out-of-core naive bayesian model fitting
"""

from sklearn.model_selection import train_test_split
from sklean.naive_bayes import GaussianNB, MultinomialNB

