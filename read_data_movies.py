import pandas as pd

def read_data(input_file="wiki_movie_plots_deduped.csv", n=1000):
    nRowsRead = n
    input_path = "dataset_movies/" + input_file
    df = pd.read_csv(input_path, delimiter=',', nrows=nRowsRead)
    
    # Filter out rows with missing plot descriptions
    df_filtered = df[df['Plot'].notna()]
    
    # Group by title and aggregate plots into lists
    grouped = df_filtered.groupby('Title')['Plot'].apply(list)
    movie_plots = grouped.to_dict()
    
    return movie_plots

def read_data_fake():
    plots = [
        "A young boy discovers he has magical powers.",
        "A group of friends embark on a journey to find a hidden treasure.",
        "A detective solves a mystery in a small town."
    ]
    result = {"Sample Movie": plots}
    return result

def merge_plots(movie_plots):
    merged_plots = []
    
    for _, plots in movie_plots.items():
        merged_plots.extend(plots)
    
    return list(merged_plots)

# Example usage:
# movie_data = read_data()
# print(len(movie_data))