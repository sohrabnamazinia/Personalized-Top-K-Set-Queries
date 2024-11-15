import pandas as pd

# Load the CSV file
df = pd.read_csv('dataset_businesses/businesses.csv')

# Iterate through rows and check for null in the 'photo_id' column
for index, row in df.iterrows():
    if pd.isnull(row['photo_id']):
        print(f"Index: {index}, Business ID: {row['business_id']}")
