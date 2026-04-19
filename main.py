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



# ==========================================
# 👤 Member 3: Ranjit
# Task: Model Training, Validation, Evaluation
# ==========================================

