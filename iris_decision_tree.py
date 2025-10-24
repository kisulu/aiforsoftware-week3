# iris_decision_tree.py
# Preprocess Iris data, train DecisionTreeClassifier, evaluate with accuracy/precision/recall


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np


# 1. Load dataset
data = load_iris(as_frame=True)
X = data.frame.drop(columns=[data.target_names.min()]) if False else data.data # safe access
# use canonical access
X = data.data
y = data.target


# 2. Handle missing values (Iris has none, but we demonstrate)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


# 3. Encode labels (already numeric for sklearn's iris)
# If textual labels were present: le = LabelEncoder(); y = le.fit_transform(y_text)


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)


# 5. Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# 6. Predict
y_pred = clf.predict(X_test)


# 7. Evaluate
acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')


print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {prec_macro:.4f}")
print(f"Recall (macro): {recall_macro:.4f}\n")
print("Classification report:\n", classification_report(y_test, y_pred, target_names=data.target_names))


# Notes: For production you'd persist the pipeline (imputer + model) using joblib or sklearn-pipeline.