"""
Below, I train logistic regression models to classify survivors on the titanic. 
I also use regularization using L2 and L1 penalties. The main goal of this file 
is to better understand Scikit-Learn's algorithm implementations.
"""

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

# Call train and test data processing from before
execfile('processing.py')
# returns train_df and test_df

###### Create Cross-Validation Dataset ######
x_train, x_test, y_train, y_test = train_test_split(train_df.drop('Survived',axis=1),train_df.Survived,
                                                     test_size=.3, random_state=0)

###### Logistic Regression ######
# Scikit-learn's logistic regression function has regularization built in. To try to
# first run a logistic regression without regularization, I set the inverse of 
# regularization strength, C, to be very high.

logistic = LogisticRegression(C=10000)
logistic.fit(x_train, y_train)
cross_val_score(logistic, x_train, y_train, cv=10).mean()
# 0.789949756784
cross_val_score(logistic, x_test, y_test, cv=10).mean()
# 0.791168091168

# submission = test_df[['PassengerId']]
# submission['Survived'] = logistic.predict(test_df)
# submission.to_csv('submissions/model6_logistic_regression.csv', index=False)
# Kaggle score of 0.74641

###### Logistic Regression with Ridge Regularization ######
# I use the default L2 penalty and try various values for C, the inverse regularization strength
l2_training_scores = []
l2_testing_scores = []
for C in np.logspace(-2, 2, 5):
    logistic_l2 = LogisticRegression(C=C, penalty='l2')
    logistic_l2.fit(x_train, y_train)
    l2_training_scores.append([C, cross_val_score(logistic_l2, x_train, y_train, cv=10).mean()])
    l2_testing_scores.append([C, cross_val_score(logistic_l2, x_test, y_test, cv=10).mean()])

sorted(l2_training_scores, key=lambda x: x[1], reverse=True)
# [[0.1, 0.79788866487455201], 
# [10.0, 0.7883368535586277], 
# [100.0, 0.7883368535586277], 
# [1.0, 0.78667354710701476], 
# [0.01, 0.71285922299027127]]
sorted(l2_testing_scores, key=lambda x: x[1], reverse=True)
# [[10.0, 0.79116809116809106], 
# [100.0, 0.79116809116809106], 
# [1.0, 0.78376068376068364], 
# [0.1, 0.76481481481481484], 
# [0.01, 0.72393162393162391]]

# C of 0.01 seems to make the regularization so strong that it harms the predictive capacity of 
# the logistic regression. C=10 or C=100 give the same score as the logistic regression without
# regularization. I will see if using C=1 has any effect on the Kaggle testing accuracy.

logistic_l2 = LogisticRegression(C=1, penalty='l2')
logistic_l2.fit(x_train, y_train)
submissionl2 = test_df[['PassengerId']]
submissionl2['Survived'] = logistic_l2.predict(test_df)
submissionl2.to_csv('submissions/model6_logistic_regression_l2.csv', index=False)
# Kaggle score of 0.74641 for C=1 (the same as for C=10000)

###### Logistic Regression with LASSO Regularization ######
# Using L1 penalty and various values of the inverse regularization cost C
l1_training_scores = []
l1_testing_scores = []
for C in np.logspace(-2, 2, 5):
    logistic_l1 = LogisticRegression(C=C, penalty='l1')
    logistic_l1.fit(x_train, y_train)
    l1_training_scores.append([C, cross_val_score(logistic_l1, x_train, y_train, cv=10).mean()])
    l1_testing_scores.append([C, cross_val_score(logistic_l1, x_test, y_test, cv=10).mean()])

sorted(l1_training_scores, key=lambda x: x[1], reverse=True)
# [[0.1, 0.79635096646185355], 
# [10.0, 0.7883368535586277], 
# [100.0, 0.7883368535586277], 
# [1.0, 0.78667354710701487], 
# [0.01, 0.64534610215053756]]
sorted(l1_testing_scores, key=lambda x: x[1], reverse=True)
# [[100.0, 0.79487179487179482], 
# [10.0, 0.79116809116809106], 
# [1.0, 0.77635327635327622], 
# [0.1, 0.77222222222222225], 
# [0.01, 0.69059829059829059]]

# Once again, very strong regularization with C=0.01 hurts the prediction. Weak LASSO regularization
# with C=100 or C=10 do about as well as when C was 10000 earlier.

logistic_l1 = LogisticRegression(C=1, penalty='l1')
logistic_l1.fit(x_train, y_train)
submissionl1 = test_df[['PassengerId']]
submissionl1['Survived'] = logistic_l1.predict(test_df)
submissionl1.to_csv('submissions/model6_logistic_regression_l1.csv', index=False)
# Kaggle score of 0.74641 for C=1 (same exact score)


# Logistic regression, logistic regression with l2, and logistic regression with l1 all 
# got the same score on Kaggle of 0.74641. Did the three logistic regressions really give
# the same predicted values?

# Differences between no regularization and L2 regularization with C=1:
len(test_df) - sum(logistic.predict(test_df) == logistic_l2.predict(test_df))
## 3
# Differences between no regularization and L1 regularization:
len(test_df) - sum(logistic.predict(test_df) == logistic_l1.predict(test_df))
## 5
# Differences between L2 and L1:
len(test_df) - sum(logistic_l2.predict(test_df) == logistic_l1.predict(test_df))
## 2

# Although the three submission received the same score, they were slightly different.
# The same score may be due to the public leaderboard only using 50% of the test set to score.
# Out of 419 observations in the test set, there are very few differences between the 
# three logistic regressions for this specific dataset. The differences between the L2
# and L1 penalties likely did not have much effect because there were not many features
# and few of the feature coefficients were heavily penalized by regularization.
