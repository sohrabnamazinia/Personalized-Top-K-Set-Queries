import os
import pandas as pd

# Input and output directories
input_dir = "Scalibility_Results"
output_dir = "Plot_scalibility"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Mapping for n values
n_mapping = {15: 500, 32: 1000, 64: 2000, 90: 4000}

# Attributes to process
attributes = [
    "total_time_init_candidates_set",
    "total_time_update_bounds",
    "total_time_compute_pdf",
    "total_time_determine_next_question",
    "total_time_llm_response"
]

# Method to include in the output
method = "Max_Prob"

# Iterate over each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        
        # Load the CSV file
        df = pd.read_csv(input_path)
        
        # Map 'n' values
        df['n'] = df['n'].map(n_mapping)
        
        for attribute in attributes:
            # Check if the required column and method are present
            if attribute in df.columns and method in df['method'].unique():
                # Filter and reshape the dataframe
                filtered_df = df[df['method'] == method][['n', attribute]]
                
                # Rename column for clarity
                filtered_df.rename(columns={attribute: method}, inplace=True)
                
                # Generate a unique output filename for the attribute
                output_filename = f"{filename.split('.csv')[0]}_{attribute}.csv"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save the output CSV
                filtered_df.to_csv(output_path, index=False)
                
                print(f"Generated: {output_filename}")
            else:
                print(f"Skipping {filename} for {attribute}: Missing required column or method.")
