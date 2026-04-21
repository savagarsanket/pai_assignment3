Titanic Survival Prediction using Logistic Regression

👥 Team Members

- Sanket Savagar - Data handling and Preprocessing
- Prajwal - Feature Engineering and encoding 
- Ranjit - Model training and Evalution
- Anurag - Reedme file

---

🎯 Problem Statement

The objective of this project is to predict whether a passenger survived the Titanic disaster or not using machine learning techniques. This is a binary classification problem where the output is either:

- 1 → Survived
- 0 → Not Survived

---

📊 Dataset Description

We used the Titanic dataset from Kaggle, which contains information about passengers such as:

- Age
- Gender
- Passenger class
- Fare
- Family details

The dataset is divided into:

- "train.csv" → used for training the model
- "test.csv" → used for testing

---

🧹 Data Preprocessing

Before training the model, we cleaned and prepared the data:

- Removed unnecessary columns:
  
  - Name
  - Ticket
  - Cabin
  - PassengerId

- Handled missing values:
  
  - Age → filled using median
  - Embarked → filled using mode
  - Fare → filled using median

---

⚙️ Feature Engineering

We created new features to improve model performance:

- FamilySize = SibSp + Parch + 1
- IsAlone:
  - 1 → passenger is alone
  - 0 → passenger has family

---

🔢 Data Encoding

Categorical variables like gender and embarked location were converted into numerical values using:

- One-hot encoding (pd.get_dummies)

This is required because machine learning models only understand numerical data.

---

✂️ Train-Test Split

The dataset was divided into:

- 80% Training data
- 20% Testing data

This helps evaluate how well the model performs on unseen data.

---

📏 Feature Scaling

We applied StandardScaler to normalize the data so that all features have similar scale, which improves model performance.

---

🤖 Model Used

We used Logistic Regression, which is suitable for classification problems.

Reasons:

- Works well for binary classification
- Simple and efficient
- Provides good accuracy

---

📊 Model Training & Validation

- Model trained using training dataset
- Applied Cross Validation (cv=5) to check consistency
- Compared training and testing accuracy

---

📈 Model Evaluation

We evaluated the model using:

- Accuracy Score → measures correctness
- Confusion Matrix → shows correct & incorrect predictions
- Classification Report:
  - Precision
  - Recall
  - F1-score

---

📉 Results

- Training Accuracy ≈ 80–85%
- Testing Accuracy ≈ 75–80%
- Cross Validation Accuracy ≈ 78–80%

👉 This indicates the model performs well and generalizes properly.

---

🔗 GitHub Collaboration

- Repository created on GitHub
- All team members were added as collaborators
- Each member contributed to different parts of the project
- Changes were managed using commits and pull requests

---

🧠 Conclusion

This project demonstrates how machine learning can be used to solve real-world classification problems.
By applying preprocessing, feature engineering, and Logistic Regression, we successfully built a model that predicts Titanic passenger survival with good accuracy.

---

🚀 Future Improvements

- Use advanced models like Random Forest or Decision Trees
- Hyperparameter tuning
- Add a web interface using Flask

---
