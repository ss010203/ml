import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")

X = pd.get_dummies(
    df.drop("Selling_Price", axis=1),
    drop_first=True,
    dtype=int
)

y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mask = (y_train >= lower_bound) & (y_train <= upper_bound)

print(mask.head())

X_train_clean = X_train[mask]
y_train_clean = y_train[mask]

print("Original training rows:", len(y_train))
print("Cleaned training rows:", len(y_train_clean))
print("Outliers removed:", len(y_train) - len(y_train_clean))

model_a = LinearRegression()
model_b = LinearRegression()

model_a.fit(X_train, y_train)
model_b.fit(X_train_clean, y_train_clean)

pred_a = model_a.predict(X_test)
pred_b = model_b.predict(X_test)

predictions = {
    "Original Data": pred_a,
    "Without Outliers": pred_b
}

for model_name, pred in predictions.items():
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)

    print(f"{model_name} - RMSE: {rmse:.2f}, R2: {r2:.2f}")