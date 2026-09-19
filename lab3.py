from statistics import correlation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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
print(f"\n--- Step 4")
MY_STUDENT_ID_LAST_4 = 1234
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=MY_STUDENT_ID_LAST_4
)
eval_model = LinearRegression()
eval_model.fit(X_train, y_train)


def evaluate_performance(model, X_data, y_data, set_name):
      preds = model.predict(X_data)
      mae = mean_absolute_error(y_data, preds)
      mse = mean_squared_error(y_data, preds)
      rmse = np.sqrt(mse)
      r2 = r2_score(y_data, preds)

      print(f"\n{set_name} indicators of the set:")
      print(f"  - MAE  : {mae:.4f}")
      print(f"  - MSE  : {mse:.4f}")
      print(f"  - RMSE : {rmse:.4f}")
      print(f"  - R²   : {r2:.4f}")


evaluate_performance(eval_model, X_train, y_train, "Train")
evaluate_performance(eval_model, X_test, y_test, "Test")
kf = KFold(n_splits=5, shuffle=True, random_state=MY_STUDENT_ID_LAST_4)
cv_r2_scores = cross_val_score(eval_model, X, y, cv=kf, scoring='r2')

print(f"\n5-fold Cross-Validation Test R² scores: {cv_r2_scores}")
print(f"Average Test R² (Mean): {cv_r2_scores.mean():.4f}")
print(f"Standard Deviation (Std): {cv_r2_scores.std():.4f}")
print("\n--- Step 6: Polynomial regression (degrees 1 to 5) ---")
degrees = [1, 2, 3, 4, 5]
train_r2_list = []
test_r2_list = []
X_single = df[[best_feature]]

for deg in degrees:
      poly_pipe = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=deg, include_bias=False),
            LinearRegression()
      )
      poly_pipe.fit(X_single, y)
      y_train_pred = poly_pipe.predict(X_single)
      train_r2 = r2_score(y, y_train_pred)
      train_r2_list.append(train_r2)
      cv_poly_scores = cross_val_score(poly_pipe, X_single, y, cv=kf, scoring='r2')
      test_r2_mean = cv_poly_scores.mean()
      test_r2_list.append(test_r2_mean)
      print(f"Degree (Degree {deg}) -> Train R²: {train_r2:.4f} | Test R² (CV): {test_r2_mean:.4f}")

plt.figure(figsize=(9, 6))
plt.plot(degrees, train_r2_list, marker='o', label='Train R²', color='blue', linewidth=2)
plt.plot(degrees, test_r2_list, marker='s', label='Test R² (5-fold CV)', color='red', linewidth=2)
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Polynomial Regression: Train vs Test R² across Degrees')
plt.xticks(degrees)
plt.legend()
plt.grid(True)
plt.show()