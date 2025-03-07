import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ds = pd.read_csv("F:/OpenCV/titanic/Titanic-Dataset.csv")

ds.loc[:, "Age"] = ds["Age"].fillna(ds["Age"].median())
ds.loc[:, "Embarked"] = ds["Embarked"].fillna(ds["Embarked"].mode()[0])


ds["Sex"] = LabelEncoder().fit_transform(ds["Sex"])

ds = pd.get_dummies(ds, columns=["Embarked"], drop_first=True)

X = ds[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]]
y = ds["Survived"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69, stratify=y)

#model
model = LogisticRegression(max_iter=1000)

#training the model
model.fit(X_train, y_train)

#predicting a random test
y_pred = model.predict(X_test)

#calculation of the metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#results
print(f"Precision: {precision:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
