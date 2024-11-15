import pandas as pd
import random
from utilities import BusinessPhoto
import os

def read_data(input_file="businesses.csv", n=1000):
    input_path = "dataset_businesses/" + input_file
    
    # Check if the file exists
    if not os.path.exists(input_path):
        print(f"Error: The file {input_path} does not exist.")
        return []

    df = pd.read_csv(input_path, delimiter=',')
    data_list = []

    sampled_rows = df.sample(n=n, replace=True)

    for _, row in sampled_rows.iterrows():
        business_info = row['info']
        photo_id = row['photo_id']
        photo_info = row['photo_info']
        
        photo_obj = BusinessPhoto(photo_id, photo_info)
        
        data_list.append((business_info, photo_obj))

    return data_list

# data = read_data(n=10000)  
# print(len(data))
