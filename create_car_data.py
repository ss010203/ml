import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

# Generate car details
year = np.random.randint(2005, 2025, n)
km_driven = np.random.randint(5000, 200000, n)

fuel = np.random.choice(
    ["Petrol", "Diesel", "CNG"],
    n,
    p=[0.55, 0.35, 0.10]
)

seller_type = np.random.choice(
    ["Individual", "Dealer"],
    n
)

transmission = np.random.choice(
    ["Manual", "Automatic"],
    n,
    p=[0.7, 0.3]
)

owner = np.random.choice(
    ["First Owner", "Second Owner", "Third Owner"],
    n,
    p=[0.65, 0.25, 0.10]
)

present_price = np.random.uniform(3, 25, n)

# Calculate vehicle age
age = 2025 - year

# Generate selling prices (in lakhs)
selling_price = (
    present_price * (0.88 ** age)
    - km_driven * 0.000008
    + np.where(fuel == "Diesel", 0.5, 0)
    + np.where(transmission == "Automatic", 0.7, 0)
    + np.where(owner == "First Owner", 0.3, -0.3)
    + np.random.normal(0, 0.7, n)
)

# Ensure positive selling prices
selling_price = np.maximum(selling_price, 0.5)

# Add a few expensive-car outliers for Q10
outlier_indices = np.random.choice(n, 15, replace=False)
selling_price[outlier_indices] *= 2.5

# Create DataFrame
df = pd.DataFrame({
    "Year": year,
    "Present_Price": present_price.round(2),
    "Kms_Driven": km_driven,
    "Fuel_Type": fuel,
    "Seller_Type": seller_type,
    "Transmission": transmission,
    "Owner": owner,
    "Selling_Price": selling_price.round(2)
})

# Save dataset
df.to_csv("used_cars.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())