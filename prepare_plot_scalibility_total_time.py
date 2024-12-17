import os
import pandas as pd

# Mapping for n values
n_mapping = {45: 1000, 101:5000, 142: 10000, 173: 15000, 201: 20000}

# Input and output folders
input_folder = "Results_Scalibility"
output_folder = "Plot_Scalibility_Ind2"
os.makedirs(output_folder, exist_ok=True)

# Process each CSV file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        # Read the input CSV file
        input_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(input_path)

        # Map n values
        df["n"] = df["n"].map(n_mapping)

        # Create a new DataFrame for the 6th chart
        output_df = df[df["method"] == "Max_Prob"][["n", "total_time"]].rename(columns={"total_time": "EntrRedInd"})

        # Add Baseline and Random columns
        baseline_df = df[df["method"] == "Exact_Baseline"][["n", "total_time"]].rename(columns={"total_time": "Baseline"})
        random_df = df[df["method"] == "Naive"][["n", "total_time"]].rename(columns={"total_time": "Random"})

        # Merge the DataFrames
        output_df = output_df.merge(baseline_df, on="n", how="outer")
        output_df = output_df.merge(random_df, on="n", how="outer")

        # Remove rows with NaN in the n column
        output_df = output_df.dropna(subset=["n"])

        # Generate output file name
        base_name, _ = os.path.splitext(file_name)
        output_file_name = f"{base_name}_6th_chart.csv"
        output_path = os.path.join(output_folder, output_file_name)

        # Save the new DataFrame to the output folder
        output_df.to_csv(output_path, index=False)

print("6th chart processed and saved.")
