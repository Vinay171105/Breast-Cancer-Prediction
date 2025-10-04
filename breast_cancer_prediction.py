import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Load the breast cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Explore the datasetp
print(df.head())
print(df.info())
print(df.size)
print(df.describe())

# Data visualization
plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=df)
plt.title('Count of Benign and Malignant Cases')
plt.show()

# Split the dataset into features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
model1 = LogisticRegression(max_iter = 10000)
model1.fit(X_train_scaled, y_train)
y_pred1 = model1.predict(X_test_scaled)
print("Accuracy Score: ", accuracy_score(y_test, y_pred1))
print("Classification Report: ", classification_report(y_test, y_pred1))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred1))
print("\n")

# K-Nearest Neighbors
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train_scaled, y_train)
y_pred2 = model2.predict(X_test_scaled)
print("Accuracy Score: ", accuracy_score(y_test, y_pred2))
print("Classification Report: ", classification_report(y_test, y_pred2))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred2))
print("\n")

# Random Forest Classifier
model3 = RandomForestClassifier(n_estimators=100, random_state=42)
model3.fit(X_train_scaled, y_train)
y_pred3 = model3.predict(X_test_scaled)
print("Accuracy Score: ", accuracy_score(y_test, y_pred3))
print("Classification Report: ", classification_report(y_test, y_pred3))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred3))

models = {
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

cv_scores = {name: np.mean(cross_val_score(model, X_train_scaled, y_train, cv=5)) for name, model in models.items()}
print("\nCross-validation scores:", cv_scores)
