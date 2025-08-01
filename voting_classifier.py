import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier

# === Load creditcard dataset ===
df = pd.read_csv("Utils/creditcard.csv")

# Separate features & target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Load pre-trained models ===
log_reg_model = joblib.load("Models/recall_logistic_model.pkl")   # recall-focused Logistic Regression
rf_model = joblib.load("Models/best_random_forest_model.pkl")    # tuned Random Forest

# === Evaluate standalone Logistic Regression ===
logreg_pred = log_reg_model.predict(X_test)
print("\n=== Standalone Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_test, logreg_pred):.4f}")
print(f"Precision: {precision_score(y_test, logreg_pred):.4f}")
print(f"Recall:    {recall_score(y_test, logreg_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, logreg_pred):.4f}")

# === Evaluate standalone Random Forest ===
rf_pred = rf_model.predict(X_test)
print("\n=== Standalone Random Forest ===")
print(f"Accuracy:  {accuracy_score(y_test, rf_pred):.4f}")
print(f"Precision: {precision_score(y_test, rf_pred):.4f}")
print(f"Recall:    {recall_score(y_test, rf_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, rf_pred):.4f}")

# === Build VotingClassifier with weights 0.2 (LogReg) and 0.8 (RF) ===
voting_clf = joblib.load("Models/final_voting_classifier.pkl")
voting_clf_pred = voting_clf.predict(X_test)

# === Evaluate Voting Classifier ===
acc = accuracy_score(y_test, voting_clf_pred)
prec = precision_score(y_test, voting_clf_pred)
rec = recall_score(y_test, voting_clf_pred)
f1 = f1_score(y_test, voting_clf_pred)

print("\n=== Voting Classifier (LogReg=0.2, RF=0.8) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# === Save final Voting Classifier ===
joblib.dump(
    [
        accuracy_score(y_test, voting_clf_pred), # 0
        recall_score(y_test, voting_clf_pred), # 1
        precision_score(y_test, voting_clf_pred), # 2
        f1_score(y_test, voting_clf_pred), # 3
        y_test, # 4
        voting_clf_pred, # 5
        X_test, # 6
    ],
    'Data/final_voting_classifier.pkl'
)
joblib.dump(voting_clf, "Models/final_voting_classifier.pkl")
print("\n Final Voting Classifier saved as 'final_voting_classifier.pkl'")
