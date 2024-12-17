import os
import pandas as pd

# Define the folder path
folder_path = "Plot_Scalibility"

# Define the mapping
value_mapping = {
    1000: "1k",
    5000: "5k",
    10000: "10k",
    15000: "15k",
    20000: "20k"
}

# Iterate over all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if the first column exists
        if df.shape[1] > 0:
            first_column_name = df.columns[0]
            
            # Replace values in the first column based on mapping
            df[first_column_name] = df[first_column_name].replace(value_mapping)
        
        # Save the updated CSV file back to the same location
        df.to_csv(file_path, index=False)

print("All files have been updated successfully.")
