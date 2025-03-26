import os
import pandas as pd
import matplotlib.pyplot as plt

base_folder = "New_Results_Fig3"
plot_csv_folder = os.path.join(base_folder, "Plot_CSV")

# List of the six CSV files we created
csv_files = [
    "Discrete_Results_Businesses_REL_Location_Around_New_York_DIV_Cost.csv",
    "Discrete_Results_Hotels_REL_Rating_of_the_hotel_DIV_Physical_distance_of_the_hotels.csv",
    "Discrete_Results_Movies_REL_Brief_plot_DIV_Different_years.csv",
    "Range_Results_Businesses_REL_Location_Around_New_York_DIV_Cost.csv",
    "Range_Results_Hotels_REL_Rating_of_the_hotel_DIV_Physical_distance_of_the_hotels.csv",
    "Range_Results_Movies_REL_Brief_plot_DIV_Different_years.csv"
]

for file_name in csv_files:
    file_path = os.path.join(plot_csv_folder, file_name)
    df = pd.read_csv(file_path)

    # Convert k to string or categorical if needed
    df['k'] = df['k'].astype(str)

    # Create a bar plot: one bar for Random, one bar for EntrRed
    plt.figure()  # Create a new figure for each CSV

    # We just plot them side by side. The simplest way is to plot them directly.
    x_positions = range(len(df['k']))
    width = 0.4

    plt.bar([x - width/2 for x in x_positions], df['Random'], width=width, label='Random')
    plt.bar([x + width/2 for x in x_positions], df['EntrRed'], width=width, label='EntrRed')

    # Configure x-ticks
    plt.xticks(x_positions, df['k'])
    plt.xlabel("k")
    plt.ylabel("API calls")
    plt.title(file_name)
    plt.legend()

    # Show the chart (or you could save it with plt.savefig)
    plt.show()
