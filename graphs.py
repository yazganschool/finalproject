import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    auc
)
from matplotlib.colors import LinearSegmentedColormap

# =====================================================
# 1. LOAD DATASET & MODELS
# =====================================================
print("[INFO] Loading dataset...")
df = pd.read_csv("creditcard.csv")  # adjust path if needed
X = df.drop("Class", axis=1)
y = df["Class"]

print("[INFO] Loading trained models...")
random_forest_model = joblib.load("best_random_forest_model.pkl")
logistic_model = joblib.load("recall_logistic_model.pkl")

# =====================================================
# 2. RANDOM FOREST EVALUATION
# =====================================================
print("[INFO] Evaluating Random Forest model...")
rfm_y_scores = random_forest_model.predict_proba(X)[:, 1]
rfm_preds = random_forest_model.predict(X)

# Precision-Recall curve + AUC
rfm_precision, rfm_recall, _ = precision_recall_curve(y, rfm_y_scores)
rfm_pr_auc = auc(rfm_recall, rfm_precision)

# Confusion matrix
rfm_cm = confusion_matrix(y, rfm_preds)
rfm_cm_normalized = rfm_cm / rfm_cm.sum()

# Feature Importances
rfm_importances = random_forest_model.feature_importances_
rfm_feature_names = X.columns
rfm_indices = np.argsort(rfm_importances)[-10:]  # Top 10 important features

# =====================================================
# 3. LOGISTIC REGRESSION EVALUATION
# =====================================================
print("[INFO] Evaluating Logistic Regression model...")
lrm_y_scores = logistic_model.predict_proba(X)[:, 1]
lrm_preds = logistic_model.predict(X)

# Precision-Recall curve + AUC
lrm_precision, lrm_recall, _ = precision_recall_curve(y, lrm_y_scores)
lrm_pr_auc = auc(lrm_recall, lrm_precision)

# Confusion matrix
lrm_cm = confusion_matrix(y, lrm_preds)
lrm_cm_normalized = lrm_cm / lrm_cm.sum()

# =====================================================
# 4. PLOTTING
# =====================================================
print("[INFO] Creating plots...")

subtitles = [
    "Feature Importance", "Confusion Matrix", "Precision-Recall Curve",
    "Feature Importance", "Confusion Matrix", "Precision-Recall Curve"
]

# 2 rows (Random Forest top, Logistic Regression bottom)
fig, axes = plt.subplots(2, 3, figsize=(22, 10))
axes = axes.flatten()

fig.text(0.51, 0.96, "Random Forest Model", ha='center', fontsize=18, weight='bold')
fig.text(0.51, 0.46, "Logistic Regression Model", ha='center', fontsize=18, weight='bold')

for ax, title in zip(axes, subtitles):
    ax.set_title(title, fontsize=14)

plt.subplots_adjust(hspace=0.4)

# ---------------- RANDOM FOREST PLOTS ---------------- #

## 1. RF Feature Importance
axes[0].barh(range(len(rfm_indices)), rfm_importances[rfm_indices],
             color='#2ec4b6', edgecolor='gray', height=0.6)
axes[0].set_yticks(range(len(rfm_indices)))
axes[0].set_yticklabels(list(rfm_feature_names[rfm_indices]), fontsize=8)
axes[0].set_xlabel('Importance Score', fontsize=12)
axes[0].set_xlim(0, max(rfm_importances[rfm_indices]) * 1.2)
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

for i, v in enumerate(rfm_importances[rfm_indices]):
    axes[0].text(v + 0.001, i, f"{v:.3f}", va='center', fontsize=8)

## 2. RF Confusion Matrix
blue_cmap = LinearSegmentedColormap.from_list("rf_blue", ["#e0f7f4", "#2ec4b6"])
sns.heatmap(
    rfm_cm_normalized,
    annot=True,
    fmt=".2%",
    cmap=blue_cmap,
    cbar=True,
    xticklabels=['Not Fraud', 'Fraud'],
    yticklabels=['Not Fraud', 'Fraud'],
    annot_kws={"fontsize": 14, "weight": "bold"},
    ax=axes[1]
)
axes[1].set_xlabel('Predicted Label', fontsize=12)
axes[1].set_ylabel('True Label', fontsize=12)

## 3. RF Precision-Recall Curve
axes[2].plot(rfm_recall, rfm_precision, color="#2ec4b6", linewidth=3,
             marker='o', alpha=0.1, markersize=5,
             label=f'PR AUC = {rfm_pr_auc:.2f}')
axes[2].set_xlabel('Recall', fontsize=12)
axes[2].set_ylabel('Precision', fontsize=12)
axes[2].legend(fontsize=12)
axes[2].grid(True, alpha=0.3)

# ---------------- LOGISTIC REGRESSION PLOTS ---------------- #

## Logistic Regression has NO feature importances
axes[3].text(0.5, 0.5, "Not Applicable\n(Logistic Regression coefficients instead)",
             ha='center', va='center', fontsize=12, color="gray")
axes[3].axis('off')

## 2. LR Confusion Matrix
green_cmap = LinearSegmentedColormap.from_list("lr_green", ["#e7f7e0", "#3bc42e"])
sns.heatmap(
    lrm_cm_normalized,
    annot=True,
    fmt=".2%",
    cmap=green_cmap,
    cbar=True,
    xticklabels=['Not Fraud', 'Fraud'],
    yticklabels=['Not Fraud', 'Fraud'],
    annot_kws={"fontsize": 14, "weight": "bold"},
    ax=axes[4]
)
axes[4].set_xlabel('Predicted Label', fontsize=12)
axes[4].set_ylabel('True Label', fontsize=12)

## 3. LR Precision-Recall Curve
axes[5].plot(lrm_recall, lrm_precision, color="#3bc42e", linewidth=3,
             marker='o', alpha=0.1, markersize=5,
             label=f'PR AUC = {lrm_pr_auc:.2f}')
axes[5].set_xlabel('Recall', fontsize=12)
axes[5].set_ylabel('Precision', fontsize=12)
axes[5].legend(fontsize=12)
axes[5].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("[INFO] Done!")
