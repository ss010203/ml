# ML Practical Q1
# Comparative Analysis of Advertising Channels
# Using Linear Regression

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Step 2: Load the dataset
df = pd.read_csv("advertising.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# Step 3: Separate independent and dependent variables
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


# Step 4: Split the dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# Step 5: Define all feature combinations
feature_sets = [
    ["TV"],
    ["Radio"],
    ["Newspaper"],
    ["TV", "Radio"],
    ["TV", "Newspaper"],
    ["Radio", "Newspaper"],
    ["TV", "Radio", "Newspaper"]
]


# Step 6: Create an empty list for results
results = []


# Step 7: Train and evaluate each combination
for features in feature_sets:

    # Select required features
    X_train_subset = X_train[features]
    X_test_subset = X_test[features]

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_subset, y_train)

    # Predict sales
    y_pred = model.predict(X_test_subset)

    # Calculate RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)

    # Store results
    results.append({
        "Features": ", ".join(features),
        "RMSE": rmse,
        "R2": r2
    })


# Step 8: Convert results into a DataFrame
results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.to_string(index=True))


# Step 9: Sort by RMSE
sorted_results = results_df.sort_values(
    by="RMSE", ascending=True
)

print("\nModels Ranked by RMSE:")
print(sorted_results.to_string(index=False))


# Step 10: Identify the best model
best_model = sorted_results.iloc[0]

print("\nBest Advertising Feature Combination:")
print("Features:", best_model["Features"])
print("RMSE:", best_model["RMSE"])
print("R2 Score:", best_model["R2"])