# ==========================================
# Titanic Survival Prediction Project (3 Members)
# ==========================================

# ==========================================
# 👤 Member 1: Sanket
# Task: Import Libraries, Load Data, Preprocessing
# ==========================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("First 5 rows:\n", train_df.head())
print("\nMissing Values:\n", train_df.isnull().sum())

# Save PassengerId
test_ids = test_df['PassengerId']

# Drop unnecessary columns
train_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
test_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

# Fill missing values
for df in [train_df, test_df]:
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

print("\nAfter Cleaning:\n", train_df.isnull().sum())


# ==========================================
# 👤 Member 2: Prajwal
# Task: Feature Engineering + Encoding + Splitting
# ==========================================
# Feature Engineering
for df in [train_df, test_df]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

# Encoding
train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

# Align columns
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

# Split data
X = train_df.drop('Survived', axis=1)
Y = train_df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 👤 Member 3: Ranjit
# Task: Model Training, Validation, Evaluation
# ==========================================


model = LogisticRegression(max_iter=2000)

# Cross validation
cv_scores = cross_val_score(model, X, Y, cv=3)

print("\nCross Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Train model
model.fit(X_train, Y_train)

train_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)

print("\nTrain Accuracy:", train_score)
print("Test Accuracy:", test_score)

# Predictions
Y_pred = model.predict(X_test)

print("\nFinal Accuracy:", accuracy_score(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

print("\nFinal Result:")
print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
print(f"Cross Validation Accuracy: {cv_scores.mean():.4f}")
