import pandas as pd
import os
import shutil

def copy_business_images(input_file="dataset_businesses/businesses.csv"):
    df = pd.read_csv(input_file)

    output_dir = "dataset_businesses/businesses_photos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

   
    for _, row in df.iterrows():
        photo_id = row['photo_id']
        photo_info = row['photo_info'] 
        
        source_image_path = f"../../yelp_photos/photos/{photo_id}"  
        
        if os.path.exists(source_image_path):
            destination_path = os.path.join(output_dir, f"{photo_id}.jpg") 

            shutil.copy(source_image_path, destination_path)
            print(f"Copied {source_image_path} to {destination_path}")
        else:
            print(f"Warning: Image {source_image_path} not found.")

    print(f"Image copying completed! All images are now in {output_dir}.")

copy_business_images()
