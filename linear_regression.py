import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset into a DataFrame
df = pd.read_csv('modified_dataset_file.csv')

# Split the data into features and labels
labels = df.iloc[:, 1:].values.reshape(-1, 12)   # rainfall values for each month
features = df.iloc[:, 0].values.reshape(-1, 1)   # year column only

# Train the linear regression model
model = LinearRegression().fit(features, labels)

# Make a prediction for a new set of features
new_features = np.array([[2023]])
prediction = model.predict(new_features)

print(prediction)
