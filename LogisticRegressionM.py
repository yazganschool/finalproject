import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Utils.ControlValues import Controls as controls
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import random
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve

# Get CSV info
creditcard = pd.read_csv('creditcard.csv')
X = creditcard.drop('Class', axis=1)
Y = creditcard['Class']


class Predictions: # All predicted / test result values from the model
    y = 0
    precision = 0 # How accurate it is
    recall = 0 # How many fraud cases did it get right
    f1 = 0 # Combination of recall and preision
class Train: # Training values for the model (80%)
    X = []
    Y = []
class Test: # Testing values for the model (20%)
    X = []
    Y = 0

# Copy X to avoid overwriting
X_scaled = X.copy()

scaler = StandardScaler()
X_scaled[['Time', 'Amount']] = scaler.fit_transform(X_scaled[['Time', 'Amount']])




'''
INFO:
Class 0 --> Real Credit Card
Class 1 --> Fake Credit Card
'''


# Get training values
Train.X, Test.X, Train.Y, Test.Y = train_test_split(
    X_scaled,
    Y,
    test_size=controls.test_size,
    random_state=controls.random_state
)


# Apply SMOTE only to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(Train.X, Train.Y)

print("After SMOTE:")
print("Training set class distribution:", dict(pd.Series(y_train_resampled).value_counts()))


# Create Model
param_grid = {
    'C': [0.005, 0.15, 1, 15],             # Regularization strength (lower = stronger regularization)
    'penalty': ['l2'],                   # Use 'l2' for ridge regression
    'solver': ['lbfgs'],                 # Use 'lbfgs' with L2 penalty
       # Try with and without class balancing
}

grid = GridSearchCV(
    estimator=LogisticRegression(class_weight=controls.class_weight, random_state=controls.random_state, max_iter=1000),
    param_grid=param_grid,
    scoring='f1',  # Focus on f1 score
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid.fit(
    X=X_train_resampled,
    y=y_train_resampled
)
rf = grid.best_estimator_

print(f"\nBest parameters found: {grid.best_params_}")
print(f"\nBest recall score during CV: {grid.best_score_}")

# Get predicted probabilities for the positive class
y_proba = rf.predict_proba(Test.X)[:, 1]

# Tune threshold
threshold = .999  # You can try different values here
Predictions.y = (y_proba >= threshold).astype(int)

# Evaluate
Predictions.precision = precision_score(Test.Y, Predictions.y)
Predictions.f1 = f1_score(Test.Y, Predictions.y)
Predictions.recall = recall_score(Test.Y, Predictions.y)

print(f"\nLogistic Regression Model Performance (Threshold={threshold}):")
print(f"Recall: {Predictions.recall:.4f}")
print(f"Precision: {Predictions.precision:.4f}")
print(f"F1 Score: {Predictions.f1:.4f}")

precisions, recalls, thresholds = precision_recall_curve(Test.Y, y_proba)
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()

plt.show()




