import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Utils/creditcard.csv')

X = df.drop('Class', axis=1)
y = df['Class']

# Scale Time and Amount
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['Time', 'Amount']] = scaler.fit_transform(X_scaled[['Time', 'Amount']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print("Training set class distribution:", dict(pd.Series(y_train_resampled).value_counts()))

model = joblib.load("Models/best_logistic_model.pkl")
# LogisticRegression(C=10, l1_ratio=1.0, max_iter=300, penalty='elasticnet',random_state=42, solver='saga')
# model.fit(X=X_train_resampled, y=y_train_resampled)

y_pred_proba = model.predict_proba(X_test)[:, 1]

# Use the same custom threshold you used before
threshold = 0.999
y_pred = (y_pred_proba >= threshold).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

dataList = [
    accuracy_score(y_test, y_pred), # 0
    recall_score(y_test, y_pred), # 1
    precision_score(y_test, y_pred), # 2
    f1_score(y_test, y_pred), # 3
    y_test, # 4
    y_pred, # 5
    X_test, # 6
]
joblib.dump(
    dataList,
    "Data/logistic_model.pkl"
)
print("Data saved in logistic_model.pkl")

#joblib.dump(model, "Models/best_logistic_model.pkl")
#i didnt delete that let me see if i cna find