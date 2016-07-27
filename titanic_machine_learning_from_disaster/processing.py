import pandas as pd

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
