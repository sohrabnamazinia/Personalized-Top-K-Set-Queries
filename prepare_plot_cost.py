import os
import pandas as pd

# Define input and output directories
input_folder = "Results_Cost"
output_folder = "Plot_Results_Cost"
os.makedirs(output_folder, exist_ok=True)

# Process each CSV file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Read the input CSV file
        df = pd.read_csv(input_path)

        # Filter and reshape the data
        filtered_df = df[df['method'].isin(['Naive', 'Max_Prob'])]
        pivot_df = filtered_df.pivot(index='k', columns='method', values='api_calls').reset_index()

        # Rename columns to match the desired output format
        pivot_df = pivot_df.rename(columns={
            'k': 'k',
            'Naive': 'Random',
            'Max_Prob': 'EntrRed'
        })

        # Save the processed data to a new CSV file
        pivot_df.to_csv(output_path, index=False)

print("Processing complete. Output saved to 'Plot_Results_Cost'.")
