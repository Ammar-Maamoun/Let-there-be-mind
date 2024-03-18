import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importing the dataset
dataset = pd.read_csv('WineQT.csv')
X = dataset.iloc[:, 0:11].values
y = dataset.iloc[:, 11].values  # Selecting only the target variable column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Define the parameter grid
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    # Add other parameters as needed
}

# Perform grid search
grid_search = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train_scaled)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Use the best model
best_regressor = grid_search.best_estimator_

# Predict using the best model on the scaled test data
y_pred_scaled = best_regressor.predict(X_test_scaled)
y_pred_scaled = y_pred_scaled.reshape(-1,1)

# Transform predictions back to the original scale
y_pred = sc_y.inverse_transform(y_pred_scaled)

# Compute metrics on the original scale
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')

# Visualize the results
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Random Forest Regression - True Values vs Predictions')
plt.legend()
plt.show()