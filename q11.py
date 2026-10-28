import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")

X = pd.get_dummies(
    df.drop("Selling_Price", axis=1),
    drop_first=True,
    dtype=int
)

y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
)

importances = importances.sort_values(ascending=False)

top_3 = importances.head(3).index

X_train_top3 = X_train[top_3]
X_test_top3 = X_test[top_3]

rf_top3 = RandomForestRegressor(random_state=42)

rf_top3.fit(X_train_top3, y_train)

pred_a = rf.predict(X_test)
pred_b = rf_top3.predict(X_test_top3)

predictions = {
    "All Features": pred_a,
    "Top 3 Features": pred_b
}

for model_name, pred in predictions.items():
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, R^2: {r2:.2f}")