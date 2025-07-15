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

# Create a custom greenish colormap
greenish_cmap = LinearSegmentedColormap.from_list("custom_greenish", ["#e0f7f4", "#2ec4b6"])

# Load data
df = pd.read_csv("creditcard.csv")
# Preprocessing (use same as training script)
X = df.drop("Class", axis=1)
y = df["Class"]


# Use same train/test split OR full data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Load best model
model = joblib.load("best_random_forest_model.pkl")

print("PARAMS:", model.get_params())

'''rf = RandomForestClassifier(
    bootstrap=True,
    ccp_alpha=0.0,
    class_weight={0: 1, 1: 10},
    criterion='gini',
    max_depth=15,
    max_features='sqrt',
    max_leaf_nodes=None,
    max_samples=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=2,
    min_samples_split=5,
    min_weight_fraction_leaf=0.0,
    n_estimators=200,
    n_jobs=None,
    oob_score=False,
    random_state=42,
    verbose=0,
    warm_start=False
)
'''

y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)

# Calculate metrics
accuracy_train = accuracy_score(y_train, y_pred_train)
recall_train = recall_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train)
f1_train = f1_score(y_train, y_pred_train)



accuracy_test = accuracy_score(y_test, y_pred)
recall_test = recall_score(y_test, y_pred)
precision_test = precision_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred)

# Evaluate with percentages
print("Accuracy TRAIN:", f"{accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
print("Recall TRAIN:", f"{recall_train:.4f} ({recall_train*100:.2f}%)")
print("Precision TRAIN:", f"{precision_train:.4f} ({precision_train*100:.2f}%)")
print("F1 Score TRAIN:", f"{f1_train:.4f} ({f1_train*100:.2f}%)")

print("Accuracy TEST:", f"{accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
print("Recall TEST:", f"{recall_test:.4f} ({recall_test*100:.2f}%)")
print("Precision TEST:", f"{precision_test:.4f} ({precision_test*100:.2f}%)")
print("F1 Score TEST:", f"{f1_test:.4f} ({f1_test*100:.2f}%)")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred, digits=4))

# --- Feature Importance Data ---
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[-10:]  # Top 10 features

# --- Confusion Matrix Data ---
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm / cm.sum()

# --- Precision-Recall Data ---
y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

# --- Create Subplots ---
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# 1. Feature Importance (Horizontal Bar)
axes[0].barh(range(len(indices)), importances[indices], color='#2ec4b6', edgecolor='gray', height=0.6)
axes[0].set_yticks(range(len(indices)))
axes[0].set_yticklabels(list(feature_names[indices]), fontsize=12)
axes[0].set_xlabel('Importance Score', fontsize=13)
axes[0].set_title('Top 10 Most Important Features', fontsize=16, pad=15)
for i, v in enumerate(importances[indices]):
    axes[0].text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=12)
axes[0].set_xlim(0, max(importances[indices]) * 1.2)
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

# 2. Confusion Matrix (Percentages)
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2%",
    cmap=greenish_cmap,
    cbar=True,
    xticklabels=['Not Fraud', 'Fraud'],
    yticklabels=['Not Fraud', 'Fraud'],
    annot_kws={"fontsize": 16, "weight": "bold"},
    ax=axes[1]
)
axes[1].set_xlabel('Predicted Label', fontsize=14)
axes[1].set_ylabel('True Label', fontsize=14)
axes[1].set_title('Confusion Matrix (Percentages)', fontsize=16, pad=15)

# 3. Precision-Recall Curve
axes[2].plot(recall, precision, color="#2ec4b6", linewidth=3, marker='o', markersize=7, label=f'PR AUC = {pr_auc:.2f}')
axes[2].set_xlabel('Recall', fontsize=13)
axes[2].set_ylabel('Precision', fontsize=13)
axes[2].set_title('Precision-Recall Curve', fontsize=16, pad=15)
axes[2].legend(fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


