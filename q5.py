import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("advertising.csv")

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

train_sizes = [100, 140, 160]

for train_size in train_sizes:
    X_train = X_pool.sample(n=train_size, random_state=4)
    y_train = y_pool.loc[X_train.index]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"Train size: {train_size}, RMSE: {rmse}, R2: {r2}")