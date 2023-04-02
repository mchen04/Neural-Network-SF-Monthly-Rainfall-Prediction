import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the dataset into a DataFrame
df = pd.read_csv('modified_dataset_file.csv')

# Split the data into features and labels
labels = df.iloc[:, 1:].values
features = df.iloc[:, 0].values.reshape(-1, 1)

# Calculate the train size as the largest multiple of 12 that is less than or equal to 80% of the length of the dataset
train_size = len(df) - len(df) % 12
train_size = int(0.8 * train_size)

# Split the data into training and test sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, shuffle=False)

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=1),
    tf.keras.layers.Dense(12, activation='linear')
])

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model on the training set
history = model.fit(train_features, train_labels, epochs=100, batch_size=32, verbose=0)

# Evaluate the model on the test set
test_predictions = model.predict(test_features)

# Compute the mean squared error between the predicted and actual rainfall values for each month
mse = mean_squared_error(test_labels, test_predictions, multioutput='raw_values')
for month, error in enumerate(mse):
    print(f"Mean Squared Error for Month {month + 1}: {error:.2f} inches")

# Create a figure with subplots for each month
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))

# Loop over each month and create a scatter plot on the corresponding subplot
for month in range(12):
    row = month // 4
    col = month % 4
    ax = axs[row][col]
    ax.plot(test_features, test_labels[:, month], label='Actual', color='blue')
    ax.plot(test_features, test_predictions[:, month], label='Predicted', color='orange')
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (inches)")
    ax.set_title(f"Month {month + 1}")
    ax.legend()

# Adjust the spacing between the subplots
plt.subplots_adjust(wspace=0.3, hspace=0.5)

# Show the figure
plt.show()
