import pandas as pd 

# def read_data(input_file="hotels.csv", n=1000):
#     nRowsRead = n 
#     input_path = "dataset_hotels/" + input_file
#     df = pd.read_csv(input_path, delimiter=',', nrows=nRowsRead, encoding='ISO-8859-1')
#     df.columns = df.columns.str.strip()
    
#     df_filtered = df[df['Description'].notnull()]
#     grouped = df_filtered.groupby('HotelName')['Description'].apply(list)
#     descriptions = grouped.to_dict()
    
#     return descriptions

def read_data(input_file="hotels.csv", n=1000):
    input_path = "dataset_hotels/" + input_file
    df = pd.read_csv(input_path, delimiter=',', encoding='ISO-8859-1')
    df.columns = df.columns.str.strip()
    
    df_filtered = df[df['Description'].notnull()]
    
    unique_hotels = set()
    indices = []
    for index, row in df_filtered.iterrows():
        hotel_name = row['HotelName']
        if hotel_name not in unique_hotels:
            unique_hotels.add(hotel_name)
        indices.append(index)
        if len(unique_hotels) >= n:
            break
    
    df_limited = df_filtered.loc[indices]
    grouped = df_limited.groupby('HotelName')['Description'].apply(list)
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
