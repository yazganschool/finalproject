import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import joblib

# Load dataset
print("Hello World")
df = pd.read_csv('Utils/creditcard.csv')
print("Dataset loaded successfully!")

# Features and labels
X = df.drop('Class', axis=1)
y = df['Class']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter tuning with GridSearchCV
print("Starting hyperparameter tuning...")

# Define the parameter grid
param_grid = {
    'class_weight': [{0:1, 1:10}, 'balanced'],  # 2 options
    'max_depth': [7, 15],                      # 2 options
    'n_estimators': [100, 200],                # 2 options
    'min_samples_split': [2, 5],               # 2 options
    'min_samples_leaf': [1, 2],                # 2 options
    'max_features': ['sqrt'],                  # 1 option
    'bootstrap': [True],                       # 1 option
    'criterion': ['gini']                      # 1 option
}


# Create base Random Forest model
base_rf = RandomForestClassifier(random_state=42)

# Create GridSearchCV object
# Using F1 score as the scoring metric since it's important for imbalanced datasets
grid_search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    scoring='f1',  # Optimize for F1 score
    cv=3,          # 3-fold cross-validation
    n_jobs=-1,     # Use all available CPU cores
    verbose=3      # Show progress
)

# Fit the grid search
print("Fitting GridSearchCV (this may take a while)...")
grid_search.fit(X_train, y_train)

# Get the best parameters and score
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.4f}")

# Use the best model for predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Results
print("\nBest Random Forest Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# Show top 5 best parameter combinations
print("\nTop 5 parameter combinations:")
cv_results = grid_search.cv_results_
top_indices = cv_results['rank_test_score'][:5]
for i, rank in enumerate(top_indices):
    idx = list(cv_results['rank_test_score']).index(rank)
    print(f"\nRank {i+1}:")
    print(f"F1 Score: {cv_results['mean_test_score'][idx]:.4f}")
    print(f"Parameters: {cv_results['params'][idx]}")

# Save the best model using joblib
print("\nSaving the best model...")
joblib.dump(best_model, 'Models/best_random_forest_model.pkl')