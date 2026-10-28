import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("used_cars.csv")

X = pd.get_dummies(
    df.drop("Selling_Price", axis=1),
    drop_first=True,
    dtype=int
)

y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(random_state=42)

param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV RMSE:", -grid.best_score_)

default_model = DecisionTreeRegressor(random_state=42)
default_model.fit(X_train, y_train)

default_pred = default_model.predict(X_test)

tuned_model = grid.best_estimator_
tuned_pred = tuned_model.predict(X_test)

predictions = {
    "Default Decision Tree": default_pred,
    "Tuned Decision Tree": tuned_pred
}

for model_name, pred in predictions.items():
    mse = mean_squared_error(y_test, pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, R^2: {r2:.2f}")