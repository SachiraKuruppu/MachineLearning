'''
Processing Titanic dataset of Kaggle
'''


import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score


# import train data
train_df = pd.read_csv("train.csv")

#train_df = (train_df - train_df.mean()) / (train_df.max() - train_df.min())

print(train_df.describe())
print("\n-----------------------------\n")

print(train_df.info())

features = train_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
print("\n-----------------------------\n")

print(features.info())
print("\n-----------------------------\n")

# Fill missing ages with mean
features["Age"].fillna(features["Age"].mean(), inplace = True)

# Fill missing embarked with most frequent
print(features["Embarked"].value_counts())
features["Embarked"].fillna("S", inplace = True)

# Change male to 1 and female to 0
features["Sex"][features["Sex"] == "male"] = 1
features["Sex"][features["Sex"] == "female"] = 0

# Change embarked to numerical values
features["Embarked"][features["Embarked"] == "S"] = 1
features["Embarked"][features["Embarked"] == "C"] = 2
features["Embarked"][features["Embarked"] == "Q"] = 3

print(features.head())
print("\n-----------------------------\n")

# Get targets
target = train_df["Survived"].values

print(target)
print("\n-----------------------------\n")

# Add family size feature
features["family"] = features["SibSp"] + features["Parch"] + 1

# Normalize everything
norm_features = (features - features.mean()) / (features.max() - features.min())

print(norm_features.describe())
print("\n-----------------------------\n")

'''
for i in range(1, 20):
    print(i)
    # Make decision tree model
    clf = DecisionTreeClassifier(max_depth=i, min_samples_split=5)
    clf.fit(features.values, target)
    
    #print(features.isnull().sum())
    
    # Check score
    scores = cross_val_score(clf, features.values, target, cv = 100)
    
    # Calculate Accuracy
    print("Accuracy = %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
    print("\n-------------------------------\n")
'''

''' Preprare Test Data '''
print("Preparing Test Data")

# import test data
test_df = pd.read_csv("test.csv")

print(test_df.describe())
print("\n-----------------------------\n")

print(test_df.info())
print("\n-----------------------------\n")

print(test_df.isnull().sum())
print("\n-----------------------------\n")

# Get test Features
test_features = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

# Add family feature
test_features["family"] = test_features["SibSp"] + test_features["Parch"] + 1

# Fill missing values
test_features["Age"].fillna(features["Age"].mean(), inplace = True)
test_features["Fare"].fillna(features["Fare"].mean(), inplace = True)

print(test_df.isnull().sum())
print("\n-----------------------------\n")

# Change male to 1 and female to 0
test_features["Sex"][test_features["Sex"] == "male"] = 1
test_features["Sex"][test_features["Sex"] == "female"] = 0

# Change embarked to numerical values
test_features["Embarked"][test_features["Embarked"] == "S"] = 1
test_features["Embarked"][test_features["Embarked"] == "C"] = 2
test_features["Embarked"][test_features["Embarked"] == "Q"] = 3

# Normalize everything
norm_test_features = (test_features - features.mean()) / (features.max() - features.min())

print(norm_test_features.head())
print("\n-----------------------------\n")

# Setting up decision tree
clf = DecisionTreeClassifier(max_depth=12, min_samples_split=5)
clf.fit(norm_features.values, target)

PassengerId = np.array(test_df["PassengerId"]).astype(int)
predictions = clf.predict(norm_test_features.values)

my_solution = pd.DataFrame(predictions, PassengerId, columns = ["Survived"])
my_solution.to_csv("my_solution.csv", index_label=["PassengerId"])



