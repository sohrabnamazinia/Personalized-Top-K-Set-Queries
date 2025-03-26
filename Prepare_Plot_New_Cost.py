import os
import pandas as pd

base_folder = 'New_Results_Fig3'
discrete_folder = os.path.join(base_folder, 'Discrete')
range_folder = os.path.join(base_folder, 'Range')
output_folder = os.path.join(base_folder, 'Plot_CSV')

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# CSV files in each folder
discrete_files = [
    "Results_Businesses_REL_Location_Around_New_York_DIV_Cost.csv",
    "Results_Hotels_REL_Rating_of_the_hotel_DIV_Physical_distance_of_the_hotels.csv",
    "Results_Movies_REL_Brief_plot_DIV_Different_years.csv"
]
range_files = [
    "Results_Businesses_REL_Location_Around_New_York_DIV_Cost.csv",
    "Results_Hotels_REL_Rating_of_the_hotel_DIV_Physical_distance_of_the_hotels.csv",
    "Results_Movies_REL_Brief_plot_DIV_Different_years.csv"
]

def process_csv(folder_name, file_name):
    file_path = os.path.join(base_folder, folder_name, file_name)
    df = pd.read_csv(file_path)
    # Keep only rows where method is Max_Prob or Naive
    df = df[df['method'].isin(['Max_Prob','Naive'])]
    # Pivot so we have columns for EntrRed (Max_Prob) and Random (Naive)
    pivoted = df.pivot(index='k', columns='method', values='api_calls').reset_index()
    pivoted.rename(columns={'Max_Prob':'EntrRed','Naive':'Random'}, inplace=True)
    # Keep only k, EntrRed, Random columns
    pivoted = pivoted[['k','EntrRed','Random']]
    
    new_name = f"{folder_name}_{file_name}"
    pivoted.to_csv(os.path.join(output_folder, new_name), index=False)

for f in discrete_files:
    process_csv('Discrete', f)

for f in range_files:
    process_csv('Range', f)
