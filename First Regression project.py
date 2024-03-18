import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('housing.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
feature_index = 1  # Change this index based on your dataset
X_test_feature = X_test[:, feature_index]

# Sort the values for better visualization
sorted_indices = np.argsort(X_test_feature)
X_test_feature_sorted = X_test_feature[sorted_indices]
y_test_sorted = y_test[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# Visualization
plt.scatter(X_test_feature_sorted, y_test_sorted, color='black', label='True Values')
plt.scatter(X_test_feature_sorted, y_pred_sorted, color='blue', label='Predictions')
plt.title('Polynomial Regression - True Values vs Predictions')
plt.xlabel(f'Feature {feature_index}')
plt.ylabel('Target Variable')
plt.legend()
plt.show()