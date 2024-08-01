import pandas as pd 

def read_data(input_file="7282_1.csv", n=1000):
    nRowsRead = n 
    input_path = "dataset_hotel_reviews/" + input_file
    df = pd.read_csv(input_path, delimiter=',', nrows=nRowsRead)
    df_filtered = df[df['reviews.username'] != 'write a review']
    grouped = df_filtered.groupby('name')['reviews.text'].apply(list)
    reviews = grouped.to_dict()
    # total_hotels = len(reviews)
    # total_reviews = sum(len(review_list) for review_list in reviews.values())
    return reviews

# r = read_data()
# print(r.keys())