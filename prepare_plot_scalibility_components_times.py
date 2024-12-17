import os
import pandas as pd

# Mapping for n values
n_mapping = {45: 1000, 101:5000, 142: 10000, 173: 15000, 201: 20000}

# Input and output folders
input_folder = "Results_Scalibility"
output_folder = "Plot_Scalibility_Ind_2"
os.makedirs(output_folder, exist_ok=True)

# Columns to process for EntrRed values
columns_to_process = [
    "total_time_init_candidates_set",
    "total_time_update_bounds",
    "total_time_compute_pdf",
    "total_time_determine_next_question",
    "total_time_llm_response"
]

# Process each CSV file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        # Read the input CSV file
        input_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(input_path)

        # Filter for Max_Prob method only
        df = df[df["method"] == "Max_Prob"]

        # Map n values
        df["n"] = df["n"].map(n_mapping)

        # Generate five output CSVs
        for col in columns_to_process:
            # Create a new DataFrame with n and EntrRed
            output_df = df[["n", col]].rename(columns={col: f"EntrRedInd_{col}"})

            # Remove rows with NaN in the column
            output_df = output_df.dropna()

            # Generate output file name
            base_name, _ = os.path.splitext(file_name)
            output_file_name = f"{base_name}_{col}.csv"
            output_path = os.path.join(output_folder, output_file_name)

            # Save the new DataFrame to the output folder
            output_df.to_csv(output_path, index=False)

print("First five charts processed and saved.")
