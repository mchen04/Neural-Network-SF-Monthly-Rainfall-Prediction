import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('SF_Rainfall.csv')

# Drop the last column
df = df.drop(df.columns[-1], axis=1)
df = df.drop(df.columns[1:3], axis=1)
df = df[['FirstYear', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]

# Create a new row with the desired values set to "no value"
new_row = {'FirstYear': 2023, 'Jan': '', 'Feb': '', 'Mar': '', 'Apr': '', 'May': '', 'Jun': '', 'Jul': '', 'Aug': '', 'Sep': '', 'Oct': '', 'Nov': '', 'Dec': ''}

# Append the new row to the DataFrame
df = df.append(new_row, ignore_index=True)

df[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']] = df[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']].shift(1)


# Output the modified DataFrame to a new CSV file
df.to_csv('modified_dataset_file.csv', index=False)
