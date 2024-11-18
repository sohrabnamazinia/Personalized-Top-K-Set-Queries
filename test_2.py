import os
import pandas as pd

# Define the path to the directory containing the CSV files
directory_path = '../../MGT Results/'

# Iterate through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        
        # Read the CSV file
        try:
            df = pd.read_csv(file_path)
            print(f"\nHeaders and last row for file: {filename}")
            
            # Print headers
            print("Headers:")
            print(df.columns.tolist())
            
            # Print the last row
            print("Last row:")
            print(df.iloc[-1].to_dict())
        except Exception as e:
            print(f"Error reading {filename}: {e}")
