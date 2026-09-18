from statistics import correlation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=pd.read_csv("data/cars_fuel_efficiency.csv")
df=df.dropna(subset=["power_hp"])
target="fuel_efficiency_km_per_l"
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix=numeric_df.corr()
print("Table Correlation Matrix")
print(correlation_matrix[target].sort_values(ascending=False))
best_feature = correlation_matrix[target].drop(target).abs().idxmax()
print(f"\nselected best feature: {best_feature}")
X = df[[best_feature]]
y = df[target]
model = LinearRegression()
model.fit(X, y)
theta_0 = model.intercept_
theta_1 = model.coef_[0]
print(f"\nIntercept (θ₀): {theta_0:.4f}")
print(f"Slope (θ₁): {theta_1:.4f}")
print(f"equation: y={theta_0:.4f}+{theta_1:.4f}*{best_feature}")
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue',alpha=0.5,label='Data points')
plt.plot(X, model.predict(X),color='red',linewidth=2,label='regression line')
plt.xlabel(best_feature)
plt.ylabel(target)
plt.title(f'Simple Linear Regression: {best_feature}vs{target}')
plt.legend()
plt.grid(True)
plt.show()
sample_value = X.mean().values[0]
predicted_efficiency = model.predict([[sample_value]])[0]
print(f"\nExample prediction: when {best_feature} = {sample_value:.2f}, "
      f"predicted fuel_efficiency = {predicted_efficiency:.2f}")
