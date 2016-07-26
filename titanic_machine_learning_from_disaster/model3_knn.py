import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# Below, I run the same processing as in model 2. 

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

##### Processing Training Data #####

# male and female
train_df['sex_boolean'] = train_df.Sex.map({'male':0, 'female':1})
train_df = train_df.drop('Sex', axis=1)

# age
median_train_age = train_df.Age.median()
train_df['Age'] = train_df['Age'].fillna(median_train_age)

# Ticket
train_df = train_df.drop('Ticket', axis=1)

# Cabin
train_df['cabin_boolean'] = 0
train_df.Cabin = train_df.Cabin.fillna(0)
train_df.loc[train_df['Cabin'] != 0,'cabin_boolean'] = 1
train_df = train_df.drop('Cabin', axis=1)

# Port of Embarkation
train_df['Embarked'] = train_df.Embarked.fillna('S')
embarked_dummies = pd.get_dummies(train_df.Embarked, prefix = 'embarked')
train_df = pd.concat([train_df, embarked_dummies], axis=1)
train_df = train_df.drop(['Embarked', 'Name'], axis=1)

##### Processing Testing Data #####

# male and female
test_df['sex_boolean'] = test_df.Sex.map({'male':0, 'female':1})
test_df = test_df.drop('Sex', axis=1)

# Age
median_test_age = test_df.Age.median()
test_df['Age'] = test_df['Age'].fillna(median_test_age)

# Tickets
test_df = test_df.drop('Ticket', axis=1)

# Fare
test_df.Fare = test_df.Fare.fillna(test_df.Fare[test_df.Pclass == 3].median())

# Cabin
test_df['cabin_boolean'] = 0
test_df.Cabin = test_df.Cabin.fillna(0)
test_df.loc[test_df['Cabin'] != 0,'cabin_boolean'] = 1
test_df = test_df.drop('Cabin', axis=1)

# Embarked
embarked_dummies = pd.get_dummies(test_df.Embarked, prefix = 'embarked')
test_df = pd.concat([test_df, embarked_dummies], axis=1)
test_df = test_df.drop(['Embarked', 'Name'], axis=1)


##### Setting up train test split #####

x_train, x_test, y_train, y_test = train_test_split(train_df.drop('Survived', axis=1),train_df.Survived,
                                                    test_size=0.2, random_state=0)

knn = neighbors.KNeighborsClassifier()

# A heuristic is to set k equal to the square root of the number of observations. Let's see what that gives us.

k = round((len(train_df) + len(test_df))**.5)
k
# 36.0

knn = neighbors.KNeighborsClassifier(n_neighbors=36, weights='uniform')
knn.fit(x_train, y_train)

knn.score(x_train, y_train)
# 0.6685393258426966

knn.score(x_test, y_test)
# 0.68715083798882681

# Since those scores are only based on one specific train/test split, finding the mean of 10-fold cross-validation 
# scores would likely hold more water...

cross_val_score(knn, x_train, y_train, cv=10).mean()
# 0.65587357478202557

cross_val_score(knn, x_test, y_test, cv=10).mean()
# 0.63137254901960793

# How likely is it that the best k is the square root of the number of obsevations? I'll try a slew of 
# k's to see which one has the best accuracy. I can also try a different weighting metric. While by 
# default, sklearn's knn classifier gives equal weight to all nearest neighbors to an observation, it 
# can also weigh neighbors proportionally to the inverse of their distances to an observation. 
# Let's see which performs better.

results = []
for weight in ['uniform', 'distance']:
    for k in xrange(1, 100):
        knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
        knn.fit(x_train, y_train)
        results.append([weight, k, cross_val_score(knn, x_train, y_train, cv=10).mean()])

sorted(results, key = lambda x: x[2], reverse=True)[:20]
# [['uniform', 34, 0.66146825396825404],
#  ['uniform', 35, 0.65873071763916835],
#  ['uniform', 38, 0.65861167002012078],
#  ['uniform', 29, 0.6573021462105969],
#  ['uniform', 33, 0.65728314330427018],
#  ['uniform', 32, 0.65728258439526044],
#  ['uniform', 22, 0.6572635814889336],
#  ['uniform', 37, 0.65724290185557788],
#  ['uniform', 36, 0.65587357478202557],
#  ['uniform', 41, 0.65581433042700643],
#  ['uniform', 27, 0.65450480661748267],
#  ['uniform', 40, 0.65442544153811755],
#  ['uniform', 31, 0.65309691482226695],
#  ['uniform', 28, 0.65305667337357465],
#  ['uniform', 45, 0.65299742901855584],
#  ['uniform', 39, 0.65299742901855562],
#  ['distance', 32, 0.65168846411804171],
#  ['distance', 43, 0.65168734630002234],
#  ['uniform', 42, 0.6515688575899844],
#  ['uniform', 19, 0.65033869885982576]]

# The best cross validated k-nearest neighbor results on the training set came from uniform weighting 
# and 34 neighbors. I will use these parameters. It's possible that the 'distance' weight gave more 
# weight to outliers, making it less successful.

knn = KNeighborsClassifier(n_neighbors=34, weights='uniform')
knn.fit(x_train, y_train)

cross_val_score(knn, x_train, y_train, cv=10).mean()
# 0.66146825396825404

cross_val_score(knn, x_test, y_test, cv=10).mean()
# 0.63692810457516347

# Looking at 34 nearest neighbors seems to do very slightly better than 36 neighbors.

knn.predict(test_df)
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       # 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       # 0, 0, 0, 0])

submission = test_df[['PassengerId']]
submission['Survived'] = knn.predict(test_df)

submission.to_csv('submissions/model3_knn.csv', index=False)

# kNN correctly predicts 0.64115 of the test set on Kaggle. This is not as good as the 
# random forest classifier, but maybe a more nuanced instance-based method will perform
# better. Stay tuned for some support vector machine models!



