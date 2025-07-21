import pandas as pd

# Define column names for the Iris dataset
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Load the data
iris = pd.read_csv('data/iris.data', names=columns)

# Drop any empty rows (if present)
iris = iris.dropna()

# Save as CSV for easier use later
iris.to_csv('data/iris.csv', index=False)

print('Iris dataset loaded and saved as data/iris.csv')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Basic statistics
print('\nBasic statistics:')
print(iris.describe())

# Pairplot
sns.pairplot(iris, hue='species')
plt.savefig('data/eda_pairplot.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(iris.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('data/eda_corr_heatmap.png')
plt.close()

print('EDA visualizations saved to data/eda_pairplot.png and data/eda_corr_heatmap.png')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Prepare data for modeling
X = iris.drop('species', axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Save the trained model
joblib.dump(clf, 'models/iris_rf_model.joblib')
print('Trained model saved to models/iris_rf_model.joblib')
