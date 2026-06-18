import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("styles.csv", on_bad_lines="skip")

# Count products by year
yearly = df.groupby("year").size().reset_index(name="ProductCount")

# Prepare training data
X = yearly[["year"]]
y = yearly["ProductCount"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Forecast years (2020–2035)
future_years = pd.DataFrame({
    "year": list(range(2020, 2036))
})

# Predict product count
future_years["PredictedCount"] = model.predict(future_years)

# Create Sales forecast
future_years["Sales"] = future_years["PredictedCount"] * 100

# Create Revenue forecast
future_years["Revenue"] = future_years["Sales"] * 1.2

# Business growth events
growth_factors = {
    2025: 1.20,  # 20% growth
    2026: 1.35,  # 35% growth
    2030: 1.25,  # 25% growth
    2031: 1.40,  # 40% growth
    2035: 1.60   # 60% growth
}

# Apply growth factors
for year, factor in growth_factors.items():
    future_years.loc[future_years["year"] == year, "Sales"] *= factor
    future_years.loc[future_years["year"] == year, "Revenue"] *= factor

# Round values
future_years["PredictedCount"] = future_years["PredictedCount"].round(0).astype(int)
future_years["Sales"] = future_years["Sales"].round(0).astype(int)
future_years["Revenue"] = future_years["Revenue"].round(0).astype(int)

# Save output
future_years.to_csv("product_forecast.csv", index=False)

print("\nForecast Created Successfully!\n")
print(future_years)