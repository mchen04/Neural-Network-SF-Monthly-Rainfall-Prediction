import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset into a DataFrame
df = pd.read_csv('modified_dataset_file.csv')

# Split the data into features and labels
#random
labels = df.iloc[:, 1:].values
features = df.iloc[:, 0].values.reshape(-1, 1)

# Calculate the train size as the largest multiple of 12 that is less than or equal to 80% of the length of the dataset
train_size = len(df) - len(df) % 12
train_size = int(0.8 * train_size)

# Split the data into training and test sets
train_features, train_labels = features[:train_size], labels[:train_size]
test_features, test_labels = features[train_size:], labels[train_size:]

# Initialize a dictionary to store the trained models for each month
models = {}

# Train a separate linear regression model for each month
for month in range(12):
    model = LinearRegression()
    model.fit(train_features, train_labels[:, month])
    models[month] = model

# Use the trained models to make predictions on the test data
test_predictions = np.zeros_like(test_labels)
for month in range(12):
    model = models[month]
    test_predictions[:, month] = model.predict(test_features)

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