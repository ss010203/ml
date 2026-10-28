import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("Social_Network_Ads.csv")

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_svm = SVC(kernel="linear")
rbf_svm = SVC(kernel="rbf")

linear_svm.fit(X_train_scaled, y_train)
rbf_svm.fit(X_train_scaled, y_train)

predictions = {
    "Linear SVM": linear_svm.predict(X_test_scaled),
    "RBF SVM": rbf_svm.predict(X_test_scaled)
}

for model_name, pred in predictions.items():
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"{model_name} - Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")


x_min = X_train_scaled[:, 0].min() - 1
x_max = X_train_scaled[:, 0].max() + 1

y_min = X_train_scaled[:, 1].min() - 1
y_max = X_train_scaled[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02),
)

grid_points = np.c_[xx.ravel(), yy.ravel()]

linear_predictions = linear_svm.predict(grid_points)
rbf_predictions = rbf_svm.predict(grid_points)

linear_predictions = linear_predictions.reshape(xx.shape)
rbf_predictions = rbf_predictions.reshape(xx.shape)

plt.figure(figsize=(12, 5))

# Linear SVM
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, linear_predictions, alpha=0.3, cmap="coolwarm")
plt.scatter(
    X_train_scaled[:, 0],
    X_train_scaled[:, 1],
    c=y_train,
    cmap="coolwarm",
    edgecolors="black"
)
plt.title("Linear SVM")
plt.xlabel("Scaled Age")
plt.ylabel("Scaled Salary")

# RBF SVM
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, rbf_predictions, alpha=0.3, cmap="coolwarm")
plt.scatter(
    X_train_scaled[:, 0],
    X_train_scaled[:, 1],
    c=y_train,
    cmap="coolwarm",
    edgecolors="black"
)
plt.title("RBF SVM")
plt.xlabel("Scaled Age")
plt.ylabel("Scaled Salary")

plt.tight_layout()
plt.show()