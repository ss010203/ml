import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")

df["Vehicle_Age"] = 2025 - df["Year"]

print(df.head())

X_a = df[["Kms_Driven"]]
X_b = df[["Kms_Driven", "Vehicle_Age"]]

y = df["Selling_Price"]

X_a_train, X_a_test, y_train, y_test = train_test_split(X_a, y, test_size=0.2, random_state=42)
X_b_train, X_b_test, _, _ = train_test_split(X_b, y, test_size=0.2, random_state=42)

model_a = LinearRegression()
model_b = LinearRegression()

model_a.fit(X_a_train, y_train)
model_b.fit(X_b_train, y_train)

pred_a = model_a.predict(X_a_test)
pred_b = model_b.predict(X_b_test)

predictions = {
    "Kms_Driven": pred_a,
    "Kms_Driven + Vehicle_Age": pred_b
}

for name, pred in predictions.items():
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5

    r2 = r2_score(y_test, pred)

    print(name, "RMSE:", rmse, "R2 score:", r2)