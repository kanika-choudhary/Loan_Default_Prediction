import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('Default_Fin.csv') 

print("Dataset Overview:")
print(data.head())
print("\nDataset Information:")
data.info()
print("\nSummary Statistics:")
print(data.describe())

data.fillna(data.median(), inplace=True)

data = pd.get_dummies(data, drop_first=True)

X = data.drop('loan_default', axis=1)  
y = data['loan_default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
print("\nFeature Importances:")
print(feature_importances)

import joblib
joblib.dump(model, 'loan_default_model.pkl')
print("Model saved as 'loan_default_model.pkl'")
