import os
import pandas as pd

# Folder path
folder_path = "Plot_Scalibility"

# Mapping dictionary
reverse_mapping = {
    '1k': 1000,
    '5k': 5000,
    '10k': 10000,
    '15k': 15000,
    '20k': 20000
}

# Go through all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Check and replace values in the first column
        df.iloc[:, 0] = df.iloc[:, 0].replace(reverse_mapping)

        # Save back to the same file
        df.to_csv(file_path, index=False)

print("Reverted '1k', '5k', etc. back to numerical values in all CSV files.")
