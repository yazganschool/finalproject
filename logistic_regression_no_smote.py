import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === Load dataset ===
df = pd.read_csv("creditcard.csv")

# Features & target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Small fast GridSearch ===
param_grid = {
    'C': [0.1, 0.5, 1, 2, 5, 10],
    'class_weight': [None, 'balanced'],
    'solver': ['lbfgs', 'liblinear'],
    'penalty': ['l2'],
    'max_iter': [500, 1000]
}



log_reg = LogisticRegression()

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='recall',
    cv=3,
    n_jobs=-1,
    verbose=2
)

print("=== Running fast GridSearchCV ===")
grid_search.fit(X_train, y_train)

print("\nBest Params:", grid_search.best_params_)
print("Best F1 score during CV:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# === Evaluate on test set ===
y_pred = best_model.predict(X_test)

print("\n=== Logistic Regression Performance on Test Set ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# ✅ Save using joblib but keep .pkl extension
dump(best_model, "recall_logistic_model.pkl")

# ✅ Later, load it like this
baseline_model = load("recall_logistic_model.pkl")

# ✅ Quick sanity check after reload
y_pred = baseline_model.predict(X_test)
print("Baseline F1 after reload:", f1_score(y_test, y_pred))
print("Baseline Recall after reload:", recall_score(y_test, y_pred))
print("Baseline Precision after reload:", precision_score(y_test, y_pred))
print("Baseline Accuracy after reload:", accuracy_score(y_test, y_pred))