import pandas as pd

# Load the dataset into a DataFrame
df = pd.read_csv('SF_Rainfall.csv')

# Drop the last column and select desired columns
df = df.drop(df.columns[-1], axis=1)
df = df.drop(df.columns[1:3], axis=1)
df = df[['FirstYear', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]

# Create a new DataFrame with the desired values set to "no value"
new_row = pd.DataFrame({'FirstYear': [2023], 'Jan': [''], 'Feb': [''], 'Mar': [''], 'Apr': [''], 'May': [''], 'Jun': [''], 'Jul': [''], 'Aug': [''], 'Sep': [''], 'Oct': [''], 'Nov': [''], 'Dec': ['']})

# Concatenate the new DataFrame to the original DataFrame
df = pd.concat([df, new_row], ignore_index=True)

# Shift the data in the columns "Jan" to "Jun" up by one row
df[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']] = df[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']].shift(1)

# Drop the second row and the last row
df = df.drop([0, len(df)-1])

# Output the modified DataFrame to a new CSV file
df.to_csv('modified_dataset_file.csv', index=False)
