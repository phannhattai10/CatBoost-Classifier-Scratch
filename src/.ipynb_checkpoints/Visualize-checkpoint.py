import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)


with open("data/processed/catboost_results.pkl","rb") as f:
    results = pickle.load(f)


y_true = np.array(results["y_true"])
y_pred = np.array(results["y_pred"])
y_prob = np.array(results["y_prob"])


accuracy = accuracy_score(y_true,y_pred)
precision = precision_score(y_true,y_pred,zero_division=0)
recall = recall_score(y_true,y_pred,zero_division=0)
f1 = f1_score(y_true,y_pred,zero_division=0)


print("Accuracy:",accuracy)
print("Precision:",precision)
print("Recall:",recall)
print("F1:",f1)


cm = confusion_matrix(y_true,y_pred)

plt.figure(figsize=(6,5))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0,1],["No Churn","Churn"])
plt.yticks([0,1],["No Churn","Churn"])

for i in range(2):
    for j in range(2):
        plt.text(j,i,str(cm[i,j]),ha="center",va="center")

plt.colorbar()
plt.tight_layout()
plt.show()


fpr,tpr,thresholds = roc_curve(y_true,y_prob)
auc = roc_auc_score(y_true,y_prob)

plt.figure(figsize=(7,5))
plt.plot(fpr,tpr,label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


precision_curve,recall_curve,thresholds = precision_recall_curve(
    y_true,
    y_prob
)

plt.figure(figsize=(7,5))
plt.plot(recall_curve,precision_curve)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()

plt.tight_layout()
plt.show()


plt.figure(figsize=(7,5))

plt.hist(
    y_prob[y_true == 0],
    bins=20,
    alpha=0.6,
    label="No Churn"
)

plt.hist(
    y_prob[y_true == 1],
    bins=20,
    alpha=0.6,
    label="Churn"
)

plt.xlabel("Predicted Probability")
plt.ylabel("Number of Customers")
plt.title("Predicted Probability Distribution")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


metrics = [
    accuracy,
    precision,
    recall,
    f1
]

names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

plt.figure(figsize=(8,5))
plt.bar(names,metrics)

plt.ylim(0,1)
plt.ylabel("Score")
plt.title("Classification Metrics")

for i,value in enumerate(metrics):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.grid(axis="y")
plt.tight_layout()
plt.show()


print("ROC-AUC:",auc)