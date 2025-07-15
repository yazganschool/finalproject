import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import warnings

import os
import sys

# Suppress all warnings globally
warnings.filterwarnings("ignore")

# Also suppress stderr output (where some sklearn warnings get printed)
class DevNull:
    def write(self, msg): pass
    def flush(self): pass

sys.stderr = DevNull()


warnings.filterwarnings("ignore")

# Load data
creditcard = pd.read_csv('creditcard.csv')
X = creditcard.drop('Class', axis=1)
Y = creditcard['Class']

# Scale Time and Amount
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['Time', 'Amount']] = scaler.fit_transform(X_scaled[['Time', 'Amount']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print("Training set class distribution:", dict(pd.Series(y_train_resampled).value_counts()))

# Safe and fast hyperparameter space
param_distributions = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['saga'],  # supports l1, l2, elasticnet
    'max_iter': [300],
    'class_weight': [None, 'balanced'],
    'l1_ratio': [0.0, 0.5, 1.0]  # only for elasticnet
}

# Randomized search
search = RandomizedSearchCV(
    LogisticRegression(random_state=42),
    param_distributions=param_distributions,
    n_iter=30,  # << keeps runtime fast
    scoring='f1',
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train_resampled, y_train_resampled)
rf = search.best_estimator_

print(f"\nBest parameters found: {search.best_params_}")
print(f"\nBest F1 score during CV: {search.best_score_}")

# Predict on test set
y_proba = rf.predict_proba(X_test)[:, 1]
threshold = 0.999
y_pred = (y_proba >= threshold).astype(int)

# Evaluation
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nModel Performance (Threshold={threshold}):")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot PR curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()

# Save model
joblib.dump(rf, 'best_logistic_model.pkl')
print("\nSaved model to 'best_logistic_model.pkl'")

plt.show()
