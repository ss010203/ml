import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

pred_linear = linear_svm.predict(X_test_scaled)
pred_rbf = rbf_svm.predict(X_test_scaled)

predictions = {
    "Linear SVM": pred_linear,
    "RBF SVM": pred_rbf
}

for name, pred in predictions.items():
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"{name} - Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

    cm = confusion_matrix(y_test, pred)

    print("Confusion Matrix:")
    print(cm)

    print()

    tn, fp, fn, tp = cm.ravel()

    print("Total Errors:", fp + fn)

    if fp > fn:
        print("More False Positives")
    elif fn > fp:
        print("More False Negatives")
    else:
        print("Equal False Positives and False Negatives")

    print()
    print()