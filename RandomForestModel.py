import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from Utils.ControlValues import Controls as controls
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import random

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

'''
INFO:
Class 0 --> Real Credit Card
Class 1 --> Fake Credit Card
'''

# Get CSV info
creditcard = pd.read_csv('Utils/creditcard.csv')
X = creditcard.drop('Class', axis=1)
Y = creditcard['Class']

# Get training values
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [1, 8],
    'class_weight': ['balanced', {0: 1, 1: 10}]
}

Train.X, Test.X, Train.Y, Test.Y = train_test_split(
    X,
    Y,
    test_size=controls.test_size,
    random_state=controls.random_state
)

# Create Model
grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=controls.random_state),
    param_grid=param_grid,
    scoring='recall',
    cv=3,
    verbose=2,
    n_jobs=-1
)
grid.fit(
    X=Train.X,
    y=Train.Y
)

rf = grid.best_estimator_

print(f"\nBest parameters found: {grid.best_params_}")
print(f"\nBest recall score during CV: {grid.best_score_}")

# Predictions
Predictions.y = rf.predict(Test.X)
Predictions.precision = precision_score(Test.Y, Predictions.y)
Predictions.f1 = f1_score(Test.Y, Predictions.y)
Predictions.recall = recall_score(Test.Y, Predictions.y)
#Predictions.accuracy = accuracy_score(Y_Test, Predictions.y)

print(f"\nRandom Forest Model Performance:")
print(f"Recall: {Predictions.recall:.4f}")
print(f"Precision: {Predictions.precision:.4f}")
print(f"F1 Score: {Predictions.f1:.4f}")

fileName = "RFM-" + str(random.randint(0, 100000)) + ".txt"

print(f"Saved under file:", fileName)
with open("Results/" + fileName, "w") as f:
  f.write(f"\nRandom Forest Model Performance:\nRecall: {Predictions.recall:.4f}\nPrecision: {Predictions.precision:.4f}\nF1 Score: {Predictions.f1:.4f}")