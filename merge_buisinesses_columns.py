import pandas as pd


csv_file_path = 'businesses.csv'


output_csv_path = 'aggregated_businesses.csv'


businesses_df = pd.read_csv(csv_file_path, engine='python')


def create_info(row):
    parts = [
        f"Name: {row['name']}" if pd.notnull(row['name']) else "Name: None",
        f"Address: {row['address']}" if pd.notnull(row['address']) else "Address: None",
        f"City: {row['city']}" if pd.notnull(row['city']) else "City: None",
        f"State: {row['state']}" if pd.notnull(row['state']) else "State: None",
        f"Stars: {row['stars']}" if pd.notnull(row['stars']) else "Stars: None",
        f"Hours: {row['hours']}" if pd.notnull(row['hours']) else "Hours: None",
        f"Review: {row['review']}" if pd.notnull(row['review']) else "Review: None"
    ]
    return ". ".join(parts)


def create_photo_info(row):
    photo_parts = [
        f"Label: {row['photo_label']}" if pd.notnull(row['photo_label']) else "Label: None",
        f"Caption: {row['photo_caption']}" if pd.notnull(row['photo_caption']) else "Caption: None"
    ]
    return ". ".join(photo_parts)


businesses_df['info'] = businesses_df.apply(create_info, axis=1)
businesses_df['photo_info'] = businesses_df.apply(create_photo_info, axis=1)
businesses_df['photo_id'] = businesses_df['photo_id'].apply(lambda x: f"{x}.jpg" if pd.notnull(x) else "None")


output_df = businesses_df[['business_id', 'info', 'photo_info', 'photo_id']]


output_df.to_csv(output_csv_path, index=False)

print(f"Aggregated data saved to {output_csv_path}")
