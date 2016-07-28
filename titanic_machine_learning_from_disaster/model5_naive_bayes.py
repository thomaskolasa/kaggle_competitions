"""
Using Naive Bayes to predictive survival on the Titanic. It is normally 
recommended to look at Naive Bayes early on for a quick model of the data.
"""

import pandas as pd

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# Calling my training and test data processing from my previous models
# returning train_df and test_df
execfile("processing.py")

##### Setting up train test split to cross-validate the training data #####

x_train, x_test, y_train, y_test = train_test_split(train_df.drop('Survived', axis=1),train_df.Survived,
                                                    test_size=0.2, random_state=0)
##### Gaussian Naive Bayes #####
# Gaussian Naive Bayes assumes the features follow a normal distribution
gnb = GaussianNB()
gnb.fit(x_train, y_train)
cross_val_score(gnb, x_train, y_train, cv=10)
cross_val_score(gnb, x_train, y_train, cv=10).mean()
# 0.7696898055
cross_val_score(gnb, x_test, y_test, cv=10)
cross_val_score(gnb, x_test, y_test, cv=10).mean()
# 0.777124183007

gnb.predict(test_df)

submission = test_df[['PassengerId']]
submission['Survived'] = gnb.predict(test_df)
submission.to_csv('submissions/model5_naive_bayes_gaussian.csv', index=False)
# On Kaggle, the Gaussian naive bayes had a score of 0.72727. The normal distribution might
# not be the best way to explain some of the training set's categorical variables...

bnb = BernoulliNB()
bnb.fit(x_train, y_train)
cross_val_score(bnb, x_train, y_train, cv=10)
cross_val_score(bnb, x_train, y_train, cv=10).mean()
# 0.780976972949

cross_val_score(bnb, x_test, y_test, cv=10)
cross_val_score(bnb, x_test, y_test, cv=10).mean()
# 0.783006535948

bnb.predict(test_df)
submission = test_df[['PassengerId']]
submission['Survived'] = bnb.predict(test_df)
submission.to_csv('submissions/model5_naive_bayes_bernoulli.csv', index=False)
# On Kaggle, the Bernoulli naive bayes had a score of 0.76077.
# The Bernoulli NB did better because many of the features are boolean variables. It is
# therfore more apt for a Bernoulli distribution to model these features.

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
cross_val_score(mnb, x_train, y_train, cv=10)
cross_val_score(mnb, x_train, y_train, cv=10).mean()
# 0.652976190476

cross_val_score(mnb, x_test, y_test, cv=10)
cross_val_score(mnb, x_test, y_test, cv=10).mean()
# 0.682679738562

mnb.predict(test_df)
submission = test_df[['PassengerId']]
submission['Survived'] = mnb.predict(test_df)
submission.to_csv('submissions/model5_naive_bayes_multinomial.csv', index=False)
# Kaggle score: 0.66507
# The multinomial Naive Bayes performed worse than the Bernoulli distributon. For tasks with more
# complicated categorical features and larger numbers of observations, it might perform better.

# Naive Bayes assumes independence of each of the obervations. However, each person who survived
# took up one of the few finite spaces on the lifeboats. This information could allow for
# a Bayesian model that takes passenger survival as dependent on the survival of other 
# passengers. Nevertheless, using Naive Bayes is simpler and less computationally expensive.
