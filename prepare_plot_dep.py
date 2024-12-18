import os
import pandas as pd

# Input folder path
input_folder = "Plot_Cost_DepInd"

# Process each CSV file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)
        
        # Rename the columns if they match the expected format
        if list(df.columns) == ['k', 'api_calls_dep', 'api_calls_ind']:
            df.columns = ['k', 'EntrRedDep', 'EntrRedInd']
        
        # Save the modified file back to the same location
        df.to_csv(file_path, index=False)

print("Column renaming completed for all relevant CSV files.")
