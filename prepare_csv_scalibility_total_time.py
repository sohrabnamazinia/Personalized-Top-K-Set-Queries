import os
import pandas as pd

# Input and output directories
input_dir = "Scalibility_Results"
output_dir = "Plot_scalibility"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Mapping for n values
n_mapping = {15: 500, 32: 1000, 64: 2000, 90: 4000}

# Methods to include in the output
methods = ["Exact_Baseline", "Naive", "Max_Prob"]

# Iterate over each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        
        # Load the CSV file
        df = pd.read_csv(input_path)
        
        # Map 'n' values
        df['n'] = df['n'].map(n_mapping)
        
        # Check if all required columns are present
        if "total_time" in df.columns and all(method in df['method'].unique() for method in methods):
            # Pivot the dataframe to create columns for the methods
            pivot_df = df.pivot(index='n', columns='method', values='total_time')[methods]
            
            # Reset the index to make 'n' a column
            pivot_df.reset_index(inplace=True)
            
            # Save the output CSV
            output_filename = f"{filename.split('.csv')[0]}_Total_time.csv"
            output_path = os.path.join(output_dir, output_filename)
            pivot_df.to_csv(output_path, index=False)
            
            print(f"Generated: {output_filename}")
        else:
            print(f"Skipping {filename}: Missing required columns or methods.")
