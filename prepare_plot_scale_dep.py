import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the input folders
dep_folder = 'Plot_Scalibility_Dep'
ind_folder = 'Plot_Scalibility_Ind'
output_folder = 'Plot_Scalibility'

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Get list of files in both folders
dep_files = set(os.listdir(dep_folder))
ind_files = set(os.listdir(ind_folder))

# Process each file in Plot_Scalibility_Dep
for ind_file in ind_files:
    # Full path to the ind file
    ind_file_path = os.path.join(ind_folder, ind_file)
    
    if ind_file in dep_files:
        # Load both CSVs
        dep_file_path = os.path.join(dep_folder, ind_file)
        dep_df = pd.read_csv(dep_file_path)
        ind_df = pd.read_csv(ind_file_path)

        # Merge the dataframes, keeping all n values from ind_df
        merged_df = pd.merge(ind_df, dep_df, on='n', how='left')
        
        # Save the merged dataframe to the output folder
        merged_file_path = os.path.join(output_folder, ind_file)
        merged_df.to_csv(merged_file_path, index=False)
    else:
        # If no matching file in dep, copy the file to output
        output_file_path = os.path.join(output_folder, ind_file)
        os.system(f'cp {ind_file_path} {output_file_path}')

# Generate charts for each file in the output folder
for file_name in os.listdir(output_folder):
    file_path = os.path.join(output_folder, file_name)
    df = pd.read_csv(file_path)
    
    # Check if the required columns exist
    if 'second_column' in df.columns and 'third_column' in df.columns:
        # Plotting
        plt.figure()
        df.plot(x='n', y=['second_column', 'third_column'], kind='line', title=file_name)
        
        # Save the figure in the output folder
        chart_file_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
        plt.savefig(chart_file_path)
        plt.close()
    else:
        print(f"Columns 'second_column' or 'third_column' missing in {file_name}. Skipping plot generation.")

print("Merging completed and charts generated.")
