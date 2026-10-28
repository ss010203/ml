import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

df = pd.read_csv("Social_Network_Ads.csv")

feature_sets = {
    "Age": ["Age"],
    "EstimatedSalary": ["EstimatedSalary"],
    "Age + EstimatedSalary": ["Age", "EstimatedSalary"]
}

y = df["Purchased"]

for model_name, features in feature_sets.items():
    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(kernel="rbf")

    svm.fit(X_train_scaled, y_train)

    pred = svm.predict(X_test_scaled)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(f"{model_name}")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("-" * 30)