import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

criteria = ["gini", "entropy"]

for criterion in criteria:
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=3,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nCriterion: {criterion}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(14, 7))

    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True
    )

    plt.title(f"Decision Tree using {criterion}")
    plt.show()