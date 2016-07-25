import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# For a random forest classifier to work, I need to first impute missing values and convert categorical variables to labels

##### Processing Training Data #####

# male and female
train_df['sex_boolean'] = train_df.Sex.map({'male':0, 'female':1})
train_df = train_df.drop('Sex', axis=1)

# age
train_df.Age.describe()
# count    714.000000
# mean      29.699118
# std       14.526497
# min        0.420000
# 25%       20.125000
# 50%       28.000000
# 75%       38.000000
# max       80.000000
# Name: Age, dtype: float64
# Since the age is right skewed with only a few elderly people, I will impute missing ages with the median age
median_train_age = train_df.Age.median()

train_df['Age'] = train_df['Age'].fillna(median_train_age)


# Looking at Number of Siblings/Spouses Aboard 
set(train_df.SibSp)
# {0, 1, 2, 3, 4, 5, 8}
# SibSp looks good

# Looking at Number of Parents/Children Aboard 
set(train_df.Parch)
# {0, 1, 2, 3, 4, 5, 6}
# Parch looks good

# Ticket numbers sometimes have different letters in them. I'll drop them for now
train_df = train_df.drop('Ticket', axis=1)

# Fare
train_df.Fare.isnull().sum()
# 0
# Fare looks good

# Cabin
# Number of people with cabin data
train_df.Cabin.isnull().sum()
# 687
# Since most of the people do not have cabin data, I will just create a flag that says if someone has a recorded cabin
train_df['cabin_boolean'] = 0
train_df.Cabin = train_df.Cabin.fillna(0)
train_df.loc[train_df['Cabin'] != 0,'cabin_boolean'] = 1
train_df = train_df.drop('Cabin', axis=1)

# Port of Embarkation
train_df.Embarked.isnull().sum()
# 2
# I will fill the two missing values with the most common port of embarkation
train_df.Embarked.mode()
# 'S'
train_df['Embarked'] = train_df.Embarked.fillna('S')

embarked_dummies = pd.get_dummies(train_df.Embarked, prefix = 'embarked')
train_df = pd.concat([train_df, embarked_dummies], axis=1)
train_df = train_df.drop('Embarked', axis=1)

# I will also drop passenger Names, though comparing titles like "Miss" or "Mrs." or "Dr." could be interesting
train_df = train_df.drop('Name', axis=1)


##### Processing Testing Data #####

# male and female
test_df['sex_boolean'] = test_df.Sex.map({'male':0, 'female':1})
test_df = test_df.drop('Sex', axis=1)

# Age
median_test_age = test_df.Age.median()
test_df['Age'] = test_df['Age'].fillna(median_test_age)

# Looking at Number of Siblings/Spouses Aboard 
set(test_df.SibSp)
# {0, 1, 2, 3, 4, 5, 8}
# looks good

# Looking at Number of Parents/Children Aboard 
set(test_df.Parch)
# {0, 1, 2, 3, 4, 5, 6, 9}
# Parch looks good

test_df = test_df.drop('Ticket', axis=1)

# Fare
test_df.Fare.isnull().sum()
# 1
test_df.Fare.describe()
# count    417.000000
# mean      35.627188
# std       55.907576
# min        0.000000
# 25%        7.895800
# 50%       14.454200
# 75%       31.500000
# max      512.329200
test_df.Pclass[test_df.Fare.isnull() == True]
# 152    3
# The missing fare is from 3rd class. I will impute the median fare of 3rd class
test_df.Fare = test_df.Fare.fillna(test_df.Fare[test_df.Pclass == 3].median())

# Cabin
test_df['cabin_boolean'] = 0
test_df.Cabin = test_df.Cabin.fillna(0)
test_df.loc[test_df['Cabin'] != 0,'cabin_boolean'] = 1
test_df = test_df.drop('Cabin', axis=1)

# Embarked
test_df.Embarked.isnull().sum()

embarked_dummies = pd.get_dummies(test_df.Embarked, prefix = 'embarked')
test_df = pd.concat([test_df, embarked_dummies], axis=1)
test_df = test_df.drop('Embarked', axis=1)

test_df = test_df.drop('Name', axis=1)


##### Running the Random Forest #####

x_train, x_test, y_train, y_test = train_test_split(train_df.drop('Survived', axis=1),train_df.Survived,
                                                    test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Training error
(1-rf.score(x_train, y_train))
# 0.012640449438202195

# Testing error
(1-rf.score(x_test, y_test))
# 0.16201117318435754

# Looking at feature importance
sorted(zip(x_train, rf.feature_importances_), key=lambda x: x[1], reverse=True)
# [('sex_boolean', 0.25217669667447506),
#  ('PassengerId', 0.19923751323528663),
#  ('Fare', 0.17472678500329081),
#  ('Age', 0.16868025688543553),
#  ('Pclass', 0.063221544438586977),
#  ('SibSp', 0.039806559860741816),
#  ('cabin_boolean', 0.031262699767317444),
#  ('Parch', 0.027587724241316568),
#  ('embarked_C', 0.016568580599839625),
#  ('embarked_S', 0.016154873261654653),
#  ('embarked_Q', 0.010576766032054975)]


rf.predict(test_df)
# array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1,
#        1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
#        1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
#        1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
#        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
#        0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
#        1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1,
#        1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
#        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
#        0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
#        0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
#        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
#        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
#        1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
#        1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
#        1, 0, 0, 0])

submission = test_df[['PassengerId']]
submission['Survived'] = rf.predict(test_df)

submission.to_csv('submissions/model2_random_forest.csv', index=False)

