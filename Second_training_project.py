import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('ENB2012_data.csv')
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, 8:10].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=2)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)  # Use transform, not fit_transform

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly_train, y_train)

# Predict using the polynomial regression model
y_pred = lin_reg_2.predict(X_poly_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')

# Visualization
plt.figure(figsize=(12, 6))

# Scatter plot for the first target variable
plt.subplot(1, 2, 1)
plt.scatter(y_test[:, 0], y_pred[:, 0], color='blue')
plt.plot([min(y_test[:, 0]), max(y_test[:, 0])], [min(y_test[:, 0]), max(y_test[:, 0])], color='red', linestyle='--', linewidth=2)
plt.xlabel('True Values - Target 1')
plt.ylabel('Predictions - Target 1')
plt.title('Polynomial Regression - Target 1')

# Scatter plot for the second target variable
plt.subplot(1, 2, 2)
plt.scatter(y_test[:, 1], y_pred[:, 1], color='green')
plt.plot([min(y_test[:, 1]), max(y_test[:, 1])], [min(y_test[:, 1]), max(y_test[:, 1])], color='red', linestyle='--', linewidth=2)
plt.xlabel('True Values - Target 2')
plt.ylabel('Predictions - Target 2')
plt.title('Polynomial Regression - Target 2')