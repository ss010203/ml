
import pandas as pd
import numpy as np

# Generate reproducible random data
np.random.seed(42)

n = 200

# Advertising expenditure
tv = np.random.uniform(0, 300, n)
radio = np.random.uniform(0, 50, n)
newspaper = np.random.uniform(0, 100, n)

# Generate sales based on advertising expenditure
sales = (
    3
    + 0.05 * tv
    + 0.20 * radio
    + 0.01 * newspaper
    + np.random.normal(0, 2, n)
)

# Create a DataFrame
df = pd.DataFrame({
    "TV": tv,
    "Radio": radio,
    "Newspaper": newspaper,
    "Sales": sales
})

# Save as CSV
df.to_csv("advertising.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Shape:", df.shape)
