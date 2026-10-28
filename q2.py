import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("advertising.csv")

X = ["TV", "Radio"]
y = ["Sales"]

X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.2, random_state=4)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_pred = poly_model.predict(X_test_poly)

for name, pred in [
    ("Linear", linear_pred),
    ("Polynomial", poly_pred)
]:
        mse = mean_squared_error(y_test, pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, pred)

        print(name, "RMSE:", rmse, "R2:", r2)
