import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")

print(df)

X_a = df[["Year", "Present_Price", "Kms_Driven"]]

X_b = pd.get_dummies(
    df.drop("Selling_Price", axis=1),
    drop_first=True,
    dtype=int
)

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
    "Numerical Features": pred_a,
    "Numerical + Categorical Features": pred_b
}

for model_name, pred in predictions.items():
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, R^2: {r2:.2f}")