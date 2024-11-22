import os
import pandas as pd

def generate_plot_results(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Define the desired order of methods
    method_order = ['Exact_Baseline', 'Naive', 'Max_Prob']
    
    # Process each file in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            input_path = os.path.join(input_folder, file)
            data = pd.read_csv(input_path)
            
            # Ensure the expected columns exist
            if 'method' not in data.columns or 'k' not in data.columns or 'api_calls' not in data.columns:
                raise ValueError(f"Unexpected file structure in {file}. Expected columns: 'method', 'k', 'api_calls'.")
            
            # Extract unique k values
            k_values = sorted(data['k'].unique())  # Sort k values for consistent ordering
            
            # Initialize a DataFrame for the output
            output_data = pd.DataFrame(index=k_values, columns=method_order)
            
            # Populate the DataFrame with API calls for each k and method
            for k in k_values:
                for method in method_order:
                    subset = data[(data['k'] == k) & (data['method'] == method)]
                    if not subset.empty:
                        output_data.loc[k, method] = subset['api_calls'].values[0]
            
            # Save the output CSV file
            output_file = os.path.join(output_folder, f"Plot_{file}")
            output_data.index.name = 'k'
            output_data.to_csv(output_file)

# Example usage
input_folder = "Results_Vary_K_Measure_Cost"
output_folder = "Plot_Results_K_Cost"
generate_plot_results(input_folder, output_folder)
