import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset into a DataFrame
df = pd.read_csv('modified_dataset_file.csv')

# Split the data into features and labels
labels = df.iloc[:, 1:].values.reshape(-1, 12)   # rainfall values for each month
features = df.iloc[:, 0].values.reshape(-1, 1)   # year column only

# Train the linear regression model
model = LinearRegression().fit(features, labels)

# Predict the monthly rainfall for 2023
year_2023 = np.array([2023]).reshape(-1, 1)
monthly_predictions_2023 = model.predict(year_2023).flatten()

# Print the predicted rainfall for each month in 2023
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for i, month in enumerate(months):
    print(f"{month}: {monthly_predictions_2023[i]:.2f} inches")

# Plot the predicted rainfall for each month in 2023
plt.plot(months, monthly_predictions_2023)
plt.xlabel('Month')
plt.ylabel('Rainfall (inches)')
plt.title('Predicted Rainfall for 2023')
plt.show()
