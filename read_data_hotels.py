import pandas as pd 

def read_data(input_file="hotels.csv", n=1000):
    nRowsRead = n 
    input_path = "dataset_hotels/" + input_file
    df = pd.read_csv(input_path, delimiter=',', nrows=nRowsRead, encoding='ISO-8859-1')
    df.columns = df.columns.str.strip()
    
    df_filtered = df[df['Description'].notnull()]
    grouped = df_filtered.groupby('HotelName')['Description'].apply(list)
    descriptions = grouped.to_dict()
    
    return descriptions

def merge_descriptions(hotels_descriptions):
    merged_descriptions = []
    
    for _, plots in hotels_descriptions.items():
        merged_descriptions.extend(plots)
    
    return list(merged_descriptions)

# Example test
# hotels = read_data()
# print(len(hotels))
