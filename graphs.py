from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

df = pd.read_csv("Utils/creditcard.csv")
# Preprocessing (use same as training script)
X = df.drop("Class", axis=1)
y = df["Class"]

class Models:
    logistic_model = joblib.load("Models/best_logistic_model.pkl")
    random_forest_model = joblib.load("Models/best_random_forest_model.pkl")
class Data:
    logistic_model = joblib.load("Data/logistic_model.pkl")
    random_forest_model = joblib.load("Data/random_forest_model.pkl")

subtitles = [
    "Feature Importance", "Confusion Matrix", "Precision-Recall Curve"
]

fig, axes = plt.subplots(2, 3, figsize=(22, 6))
axes = axes.flatten()

fig.text(0.51, 0.94, "Random Forest Model", ha='center', fontsize=16)
fig.text(0.51, 0.42, "Logistic Regression Model", ha='center', fontsize=16)
for ax, title in zip(axes, subtitles): ax.set_title(title)

plt.tight_layout(rect=[0, 1, 1, 1])
plt.subplots_adjust(hspace=0.8)

# Random Forest Model
rfm_y_scores = Models.random_forest_model.predict_proba(Data.random_forest_model[6])[:, 1]
rfm_precision, rfm_recall, rfm_thresholds = precision_recall_curve(Data.random_forest_model[4], rfm_y_scores)
rfm_pr_auc = auc(rfm_recall, rfm_precision)
rfm_importances = Models.random_forest_model.feature_importances_
rfm_feature_names = X.columns
rfm_indices = np.argsort(rfm_importances)[-10:]  # Top 10 features

rfm_cm = confusion_matrix(Data.random_forest_model[4], Data.random_forest_model[5])
rfm_cm_normalized = rfm_cm / rfm_cm.sum()

# // Feature Importance
axes[0].barh(range(len(rfm_indices)), rfm_importances[rfm_indices], color='#2ec4b6', edgecolor='gray', height=0.6)
axes[0].set_yticks(range(len(rfm_indices)))
axes[0].set_yticklabels(list(rfm_feature_names[rfm_indices]), fontsize=12)
axes[0].set_xlabel('Importance Score', fontsize=13)
for i, v in enumerate(rfm_importances[rfm_indices]):
    axes[0].text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=8)
axes[0].set_xlim(0, max(rfm_importances[rfm_indices]) * 1.2)
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

# // Confusion Matrix
blue_cmap = LinearSegmentedColormap.from_list("custom_greenish", ["#e0f7f4", "#2ec4b6"])
sns.heatmap(
    rfm_cm_normalized,
    annot=True,
    fmt=".2%",
    cmap=blue_cmap,
    cbar=True,
    xticklabels=['Not Fraud', 'Fraud'],
    yticklabels=['Not Fraud', 'Fraud'],
    annot_kws={"fontsize": 16, "weight": "bold"},
    ax=axes[1]
)
axes[1].set_xlabel('Predicted Label', fontsize=12)
axes[1].set_ylabel('True Label', fontsize=12)

# // Precision-Recall Graph
axes[2].plot(rfm_recall, rfm_precision, color="#2ec4b6", linewidth=3, marker='o', alpha=.1, markersize=5, label=f'PR AUC = {rfm_pr_auc:.2f}')
axes[2].set_xlabel('Recall', fontsize=13)
axes[2].set_ylabel('Precision', fontsize=13)
axes[2].legend(fontsize=12)
axes[2].grid(True, alpha=0.3)

# Logistic Regression Model

plt.show()