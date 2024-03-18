import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
dataset = pd.read_csv('yahoo_stock.csv')

# Extract the target variable
y = dataset.iloc[:, -1].values

# Normalize the target variable to the range [0, 1]
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Set sequence length (adjust as needed)
sequence_length = 60  # You can experiment with different sequence lengths

# Create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append((sequence, target))
    return np.array(sequences)

# Create sequences for training
sequences = create_sequences(y_scaled, sequence_length)

# Split data into features and target
X = np.array([seq for seq, _ in sequences])
y = np.array([target for _, target in sequences])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=5)  # You can adjust the number of components
X_pca = pca.fit_transform(X.reshape(X.shape[0], -1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()

# Add an LSTM layer
model.add(LSTM(units=100, input_shape=(X_train.shape[1], 1)))

# Add a dense output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to original scale
predictions_original_scale = scaler.inverse_transform(predictions)

# Inverse transform the true values to original scale for evaluation
y_test_original_scale = scaler.inverse_transform(y_test.reshape(-1, 1))

# Convert predictions to binary values based on a threshold (adjust threshold as needed)
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)
binary_y_test = (y_test > threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(binary_y_test, binary_predictions)
print(f'Accuracy: {accuracy}')

# Evaluate the model
mse = mean_squared_error(y_test_original_scale, predictions_original_scale)
print(f'Mean Squared Error: {mse}')