import pandas as pd
import random

# Load the CSV file into a DataFrame
df = pd.read_csv('aggregated_businesses.csv')

# Randomly sample one row
sampled_row = df.sample(n=1)

# Print the values of each column for the sampled row
for column in df.columns:
    print(f"{column}: {sampled_row[column].values[0]}")
